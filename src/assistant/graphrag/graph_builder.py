"""Сборка графа знаний из результатов извлечения. Требование FR-2."""

import networkx as nx

from assistant.infra.schemas import ExtractionResult


def build_graph(results: list[ExtractionResult]) -> nx.DiGraph:
    """Объединяет результаты по всем чанкам в один направленный граф.

    Узел: идентификатор `Entity.node_id`, атрибуты `type`, `name`, `sources` (список chunk_id).
    Ребро: атрибуты `type`, `weight` (число разных чанков, где связь найдена), `sources`.
    Связи, не соответствующие схеме (`Relation.matches_schema`), отбрасываются. Если между двумя
    деталями есть `REPLACED_BY` в обе стороны, остаётся связь с большим весом, при равенстве
    отбрасываются обе. Число таких конфликтов записывается в `graph.graph["conflicts"]`.
    """
    graph = nx.DiGraph()
    graph.graph["conflicts"] = 0
    for result in results:
        valid = [relation for relation in result.relations if relation.matches_schema()]
        entities = list(result.entities) + [end for relation in valid for end in (relation.subject, relation.object)]
        for entity in entities:
            if entity.node_id not in graph:
                graph.add_node(entity.node_id, type=entity.type, name=entity.name, sources=[])
            sources = graph.nodes[entity.node_id]["sources"]
            if result.chunk_id not in sources:
                sources.append(result.chunk_id)
        for relation in valid:
            left, right = relation.subject.node_id, relation.object.node_id
            if not graph.has_edge(left, right):
                graph.add_edge(left, right, type=relation.type, weight=0, sources=[])
            edge = graph.edges[left, right]
            if result.chunk_id not in edge["sources"]:
                edge["sources"].append(result.chunk_id)
                edge["weight"] = len(edge["sources"])

    checked: set[frozenset[str]] = set()
    for left, right, edge in list(graph.edges(data=True)):
        pair = frozenset((left, right))
        if left == right or pair in checked or edge["type"] != "REPLACED_BY":
            continue
        checked.add(pair)
        if not graph.has_edge(right, left) or graph.edges[right, left]["type"] != "REPLACED_BY":
            continue
        graph.graph["conflicts"] += 1
        forward, backward = edge["weight"], graph.edges[right, left]["weight"]
        if forward <= backward:
            graph.remove_edge(left, right)
        if backward <= forward:
            graph.remove_edge(right, left)
    for _, node in graph.nodes(data=True):
        node["sources"].sort()
    for _, _, edge in graph.edges(data=True):
        edge["sources"].sort()
    return graph
