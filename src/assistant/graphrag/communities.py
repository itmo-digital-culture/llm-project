"""Сообщества графа и их сводки. Требование FR-3."""

from collections import Counter

import networkx as nx

from assistant.infra.llm import LLMClient
from assistant.infra.schemas import Community

RELATION_MEANINGS = {
    "BELONGS_TO": "Модель {source} входит в линейку «{target}».",
    "HAS_PART": "В модели {source} установлена деталь {target}.",
    "IS_A": "Деталь {source} относится к виду компонентов «{target}».",
    "INDICATES": "Код ошибки {source} указывает на отказ компонента «{target}».",
    "FIXED_BY": "Ошибку {source} устраняют по регламенту {target}.",
    "REPLACED_BY": "Устаревшая деталь {source} заменена на деталь {target}.",
    "SUPPLIED_BY": "Деталь {source} поставляет поставщик «{target}».",
    "FAILED": "Отказ компонента «{target}» у модели {source} упомянут в отрывках документов (число: {weight}).",
}


def detect_communities(graph: nx.DiGraph) -> list[Community]:
    """Разбивает узлы на сообщества алгоритмом Лувена с фиксированным `seed`.

    Каждый узел входит ровно в одно сообщество. Поле `summary` здесь остаётся пустым.
    """
    undirected = nx.Graph()
    undirected.add_nodes_from(graph.nodes)
    for source, target, attrs in graph.edges(data=True):
        weight = attrs["weight"]
        if undirected.has_edge(source, target):
            undirected[source][target]["weight"] += weight
        else:
            undirected.add_edge(source, target, weight=weight)

    if undirected.number_of_edges() == 0:
        groups = [[node] for node in sorted(undirected)]
    else:
        partition = nx.community.louvain_communities(undirected, weight="weight", seed=42)
        groups = sorted(sorted(nodes) for nodes in partition)

    return [Community(community_id, nodes) for community_id, nodes in enumerate(groups)]


def summarize_community(community: Community, graph: nx.DiGraph, llm: LLMClient) -> str:
    """Сводка сообщества не длиннее 150 слов: какие сущности входят, как связаны, что встречается чаще.
    Все числа считает код по весам рёбер и передаёт модели готовыми. Модель их только формулирует.
    """
    subgraph = graph.subgraph(community.nodes)
    names = {node: subgraph.nodes[node].get("name", node.split(":", 1)[-1]) for node in sorted(subgraph)}
    facts = []
    failures_by_model: Counter[str] = Counter()
    failures_by_component: Counter[str] = Counter()
    for source, target, attrs in sorted(subgraph.edges(data=True)):
        relation = attrs["type"]
        weight = attrs["weight"]
        facts.append(RELATION_MEANINGS[relation].format(source=names[source], target=names[target], weight=weight))
        if relation == "FAILED":
            failures_by_model[source] += weight
            failures_by_component[target] += weight

    sections = [
        "Сущности:\n" + ("\n".join(f"- {node}" for node in names) or "нет"),
        "Факты:\n" + ("\n".join(f"- {fact}" for fact in facts) or "нет"),
    ]
    if len(failures_by_model) > 1 or len(failures_by_component) > 1:
        for title, counts in (
            ("Суммы весов FAILED по моделям", failures_by_model),
            ("Суммы весов FAILED по узлам оборудования", failures_by_component),
        ):
            rows = "\n".join(f"- {names[node]}: {weight}" for node, weight in counts.most_common())
            sections.append(f"{title}:\n{rows}")

    response = llm.chat(
        [
            {
                "role": "system",
                "content": (
                    "Перед тобой аналитика, собранная по сервисной документации об оборудовании. "
                    "Проанализируй факты, их взаимосвязи и частоту упоминаний. "
                    "Подготовь для сотрудника сервисной службы связную сводку на русском языке, до 150 слов. "
                    "Объединяй связанные сведения, например код ошибки, отказавший узел и регламент "
                    "или исходную деталь, её замену и поставщика. Выделяй главное и убирай повторы. "
                    "Сохраняй исходные коды, названия и смысл связей. Привязывай каждую деталь к указанной "
                    "для неё модели. Используй только переданные сведения. "
                    "Числа из блока «Факты» обозначают количество упоминаний в документации. "
                    "Числа из блоков «Суммы весов» служат только для выбора главного и остаются за кадром. "
                    "Формулируй выводы обычными словами, без технических меток и рекомендаций. "
                    "Сравнивай только сопоставимые факты из входных данных. К примеру для одного компонента "
                    "можно сравнить число отрывков с упоминанием его отказа у разных моделей. "
                    "Если дан один факт, передай его одним предложением, сохранив смысл связи и её участников. "
                    "Если фактов нет, кратко назови сущности."
                ),
            },
            {"role": "user", "content": "\n\n".join(sections)},
        ]
    )
    summary = response.content.strip()
    words = summary.split(maxsplit=150)
    return " ".join(words[:150]) if len(words) > 150 else summary
