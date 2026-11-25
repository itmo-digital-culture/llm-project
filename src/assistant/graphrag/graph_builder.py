"""Сборка графа знаний из результатов извлечения. Требование FR-2."""

import networkx as nx

from assistant.infra.schemas import Entity, ExtractionResult


def build_graph(results: list[ExtractionResult]) -> nx.DiGraph:
    """Объединяет результаты по всем чанкам в один направленный граф.

    Узел: идентификатор `Entity.node_id`, атрибуты `type`, `name`, `sources` (список chunk_id).
    Ребро: атрибуты `type`, `weight` (число разных чанков, где связь найдена), `sources`.
    Связи, не соответствующие схеме (`Relation.matches_schema`), отбрасываются. Если между двумя
    деталями есть `REPLACED_BY` в обе стороны, остаётся связь с большим весом, при равенстве
    отбрасываются обе. Число таких конфликтов записывается в `graph.graph["conflicts"]`.
    """

    nodes = {}
    edges = {}

    for result in results:
        for entity in result.entities:
            _add_node(nodes, entity, result.chunk_id)

        for relation in result.relations:
            if not relation.matches_schema():
                continue

            for entity in (relation.subject, relation.object):
                _add_node(nodes, entity, result.chunk_id)

            key = (relation.subject.node_id, relation.type, relation.object.node_id)
            edge = edges.setdefault(key, {"sources": set()})
            edge["sources"].add(result.chunk_id)

    conflicts = _drop_conflicts(edges)
    graph = nx.DiGraph()

    for node_id, attr in nodes.items():
        graph.add_node(node_id, type=attr["type"], name=attr["name"], sources=sorted(attr["sources"]))

    for (subject, rel_type, obj), attrs in edges.items():
        graph.add_edge(subject, obj, type=rel_type, weight=len(attrs["sources"]), sources=sorted(attrs["sources"]))

    graph.graph["conflicts"] = conflicts
    return graph


def _add_node(nodes: dict[str, dict], entity: Entity, chunk_id: str) -> None:
    node = nodes.setdefault(entity.node_id, {"type": entity.type, "name": entity.name, "sources": set()})
    node["sources"].add(chunk_id)


def _drop_conflicts(edges) -> int:
    conflicts = 0

    for key in list(edges):
        subject, rel_type, obj = key

        if rel_type != "REPLACED_BY" or subject >= obj:
            continue

        opposite = (obj, rel_type, subject)

        if opposite not in edges:
            continue

        conflicts += 1
        forward, backward = len(edges[key]["sources"]), len(edges[opposite]["sources"])

        if forward < backward:
            del edges[key]
        elif backward < forward:
            del edges[opposite]
        else:
            del edges[key]
            del edges[opposite]

    return conflicts
