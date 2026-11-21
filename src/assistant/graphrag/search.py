"""Два режима поиска по графу. Требования FR-4, FR-5 и FR-10."""

import json
from itertools import combinations, pairwise

import networkx as nx

from assistant.infra.grounding import code_type, find_codes, find_names
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import MAP_SCHEMA, Answer, KnowledgeBase

LOCAL_CONTEXT_LIMIT = 8000
NOT_FOUND = "В базе знаний не найдено."


def _local_edges(graph: nx.DiGraph, anchors: list[str]) -> set[tuple[str, str]]:
    undirected = graph.to_undirected(as_view=True)
    if len(anchors) == 1:
        neighbours = nx.single_source_shortest_path_length(undirected, anchors[0], cutoff=2)
        return set(graph.subgraph(neighbours).edges)

    edges: set[tuple[str, str]] = set()
    for start, end in combinations(anchors, 2):
        for path in nx.all_simple_paths(undirected, start, end, cutoff=3):
            for source, target in pairwise(path):
                if graph.has_edge(source, target):
                    edges.add((source, target))
                if graph.has_edge(target, source):
                    edges.add((target, source))

    # Замены направлены от старой детали к новой, в том числе за пределами путей между опорами.
    pending = {node for edge in edges for node in edge if node.startswith("Part:")}
    visited: set[str] = set()
    while pending:
        part = pending.pop()
        visited.add(part)
        for replacement, attrs in graph[part].items():
            if attrs["type"] == "REPLACED_BY":
                edges.add((part, replacement))
                if replacement not in visited:
                    pending.add(replacement)
    return edges


def local_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Отвечает на вопрос о конкретных сущностях по фактам из графа.

    Опорные сущности ищутся без модели: коды через `find_codes`, названия через `find_names`.
    При двух и более опорных сущностях берутся факты на путях длиной до трёх связей между ними
    и цепочки `REPLACED_BY` от деталей на этих путях. При одной берётся окрестность на глубину
    до двух связей. Модель получает пронумерованный список фактов и отвечает только по нему.
    Если в вопросе есть код, которого нет в графе, или фактов нет, возвращается
    `Answer(..., refused=True)` без вызова модели.
    """
    graph = kb.graph
    code_nodes = [f"{code_type(code)}:{code}" for code in find_codes(question)]
    if any(node not in graph for node in code_nodes):
        return Answer(NOT_FOUND, refused=True)

    name_nodes = (f"{kind}:{name}" for kind, name in find_names(question, kb.vocabulary))
    anchors = list(dict.fromkeys(code_nodes + [node for node in name_nodes if node in graph]))
    if not anchors:
        return Answer(NOT_FOUND, refused=True)

    facts: list[str] = []
    lines: list[str] = []
    sources: set[str] = set()
    context_length = 0
    for source, target in sorted(_local_edges(graph, anchors)):
        attrs = graph[source][target]
        source_name = graph.nodes[source]["name"]
        target_name = graph.nodes[target]["name"]
        relation = attrs["type"]
        fact = f"{source_name} -{relation}-> {target_name}"
        line = f"{len(lines) + 1}. {fact}"
        new_length = context_length + len(line) + bool(lines)
        if new_length > LOCAL_CONTEXT_LIMIT:
            continue
        facts.append(fact)
        lines.append(line)
        context_length = new_length
        sources.update(chunk_id.partition("#")[0] for chunk_id in attrs["sources"])

    if not facts:
        return Answer(NOT_FOUND, refused=True)

    context = "\n".join(lines)
    response = llm.chat(
        [
            {
                "role": "system",
                "content": (
                    "Ответь на вопрос на русском языке проанализировав предоставленные факты. "
                    "Каждый факт имеет вид «исходная сущность -ТИП_СВЯЗИ-> целевая сущность». "
                    "Значения связей для записи X -ТИП-> Y:\n"
                    "BELONGS_TO — модель X входит в линейку Y.\n"
                    "HAS_PART — в модели X установлена деталь Y.\n"
                    "IS_A — деталь X относится к виду компонентов Y.\n"
                    "INDICATES — ошибка X указывает на отказ компонента Y.\n"
                    "FIXED_BY — ошибку X устраняют по регламенту Y.\n"
                    "REPLACED_BY — деталь X заменена деталью Y.\n"
                    "SUPPLIED_BY — деталь X поставляет поставщик Y.\n"
                    "FAILED — у модели X зафиксирован отказ компонента Y.\n"
                    "Верни краткий итоговый ответ без упоминания фактов. "
                    "Ответ пользователю не должен содержать полной цепочки размышлений. "
                    "В своём ответе используй коды из фактов. "
                    f"Если данных для ответа недостаточно, ответь: «{NOT_FOUND}»"
                ),
            },
            {"role": "user", "content": f"Вопрос: {question}\n\nФакты:\n{context}"},
        ]
    )
    text = response.content.strip()
    if not text or NOT_FOUND.rstrip(".").casefold() in text.casefold():
        return Answer(NOT_FOUND, sources=sorted(sources), facts=facts, refused=True)
    return Answer(text, sources=sorted(sources), facts=facts)


def global_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Отвечает на обзорный вопрос по сводкам сообществ в два шага.

    Шаг map: для каждой сводки модель даёт частичный ответ и оценку полезности от 0 до 10.
    Шаг reduce: частичные ответы с оценкой выше 0 объединяются в итоговый ответ.
    Если таких ответов нет, возвращается `Answer(..., refused=True)`.
    """
    partial_answers: list[str] = []
    sources: set[str] = set()
    for community in kb.communities:
        response = llm.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "Сформулируй частичный ответ на русском языке только по данной сводке. "
                        "Верни JSON с полями answer и score. Оцени полезность от 0 до 10. "
                        "Если сводка не содержит сведений для ответа, верни пустой answer и score 0. "
                        "Не изменяй названия, коды, область сравнения и единицы измерения. "
                        "Включи в частичный ответ сведения из сводки, нужные для обоснования вывода."
                    ),
                },
                {"role": "user", "content": f"Вопрос: {question}\n\nСводка:\n{community.summary}"},
            ],
            json_schema=MAP_SCHEMA,
        )
        try:
            partial = json.loads(response.content)
        except json.JSONDecodeError:
            continue

        if not isinstance(partial, dict):
            continue
        text = partial.get("answer")
        score = partial.get("score")
        if not isinstance(text, str) or not text.strip() or type(score) is not int or not 1 <= score <= 10:
            continue
        partial_answers.append(text.strip())

        # Сводка опирается на внутренние связи сообщества и может упоминать узлы без таких связей.
        subgraph = kb.graph.subgraph(community.nodes)
        for _, _, attrs in subgraph.edges(data=True):
            sources.update(chunk.partition("#")[0] for chunk in attrs.get("sources", ()))
        for node in nx.isolates(subgraph):
            sources.update(chunk.partition("#")[0] for chunk in subgraph.nodes[node].get("sources", ()))

    if not partial_answers:
        return Answer(NOT_FOUND, refused=True)

    context = "\n".join(f"{index}. {text}" for index, text in enumerate(partial_answers, start=1))
    response = llm.chat(
        [
            {
                "role": "system",
                "content": (
                    "Дай краткий итоговый ответ на русском по уже отобранным частичным ответам. "
                    "Если один частичный ответ прямо отвечает на вопрос, используй его вывод. "
                    "Если ответов несколько, объедини совместимые сведения и убери повторы. "
                    "Сохраняй точные коды, названия, числа и границы сравнения. "
                    "Используй только переданные сведения."
                ),
            },
            {"role": "user", "content": f"Вопрос: {question}\n\nЧастичные ответы:\n{context}"},
        ]
    )
    text = response.content.strip()
    if not text or NOT_FOUND.rstrip(".").casefold() in text.casefold():
        return Answer(NOT_FOUND, refused=True)
    return Answer(text, sources=sorted(sources))
