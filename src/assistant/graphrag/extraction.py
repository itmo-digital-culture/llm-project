"""Извлечение сущностей и связей из чанка. Требование FR-1."""

import json

from assistant.infra.config import vocabulary
from assistant.infra.grounding import CODE_TYPES, canonical_name, find_codes, find_names
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import (
    ENTITY_TYPES,
    EXTRACTION_SCHEMA,
    RELATION_SIGNATURES,
    Chunk,
    Entity,
    ExtractionResult,
    Relation,
)

SYSTEM = (
    "Ты извлекаешь сущности и связи из фрагемнтов документов сервисной службы 'ПолярТех'. "
    "Верни только JSON-объект по схеме, без пояснений"
)

RELATION_HINTS = {
    "BELONGS_TO": "модель входит в линейку",
    "HAS_PART": "в модели установлена деталь",
    "IS_A": "деталь является узлом этого вида",
    "INDICATES": "ошибка указывает на отказ узла",
    "FIXED_BY": "ошибка устраняется по регламенту",
    "REPLACED_BY": "старая деталь снята с производства, связь ведёт от старой детали к новой",
    "SUPPLIED_BY": "деталь поставляет поставщик",
    "FAILED": "в отчёте о выезде зафиксирован отказ узла",
}

RULES = (
    "Правила:\n"
    "- Коды пиши латиницей, как во фрагменте: модели AR-210, детали P-1042B, ошибки E-17, регламенты SP-104.\n"
    "- Названия линеек, узлов и поставщиков бери только из словаря, в именительном падеже, как записаны.\n"
    "- Если сказано, что в модели установлены узлы или детали, каждую названную деталь связывай "
    "с моделью фрагмента связью HAS_PART.\n"
    "- Строка вида «узел – деталь P-XXXX, поставщик «Y»» даёт три связи: HAS_PART (модель -> деталь), "
    "IS_A (деталь -> узел), SUPPLIED_BY (деталь -> поставщик).\n"
    "- Фраза «поставщик детали P-XXXX – «Y»» — это связь SUPPLIED_BY от названной детали к поставщику.\n"
    "- Включай только то, что прямо сказано во фрагменте. Ничего не выдумывай и не переноси из примеров.\n"
    "- Сущности из связей включай и в список entities."
)

EXAMPLES = (
    (
        "# Сервисный бюллетень SB-2025-11\n\n"
        "Деталь P-2054 (нагреватель оттайки) снята с производства. Вместо неё устанавливается "
        "деталь P-2054B. Поставщик детали P-2054B – «ТермоПрибор».",
        [
            ("Part", "P-2054"),
            ("Part", "P-2054B"),
            ("Component", "нагреватель оттайки"),
            ("Supplier", "ТермоПрибор"),
        ],
        [
            ("Part", "P-2054", "REPLACED_BY", "Part", "P-2054B"),
            ("Part", "P-2054", "IS_A", "Component", "нагреватель оттайки"),
            ("Part", "P-2054B", "SUPPLIED_BY", "Supplier", "ТермоПрибор"),
        ],
    ),
    (
        "# Отчёт о выезде R-2025-042\n\n"
        "Оборудование: моноблок BZ-110. На панели горел код E-12.\n\n"
        "Диагностика показала отказ узла: компрессор.",
        [("Model", "BZ-110"), ("ErrorCode", "E-12"), ("Component", "компрессор")],
        [("Model", "BZ-110", "FAILED", "Component", "компрессор")],
    ),
    (
        "# Руководство по эксплуатации: витрина AR-420\n\n"
        "Витрина AR-420 входит в линейку «Арктика». В AR-420 установлены следующие узлы.\n\n"
        "- компрессор – деталь P-9010, поставщик «КомпрессорМаш»\n"
        "- пусковое реле – деталь P-9042, поставщик «ХолодКомплект»",
        [
            ("Model", "AR-420"),
            ("ProductLine", "Арктика"),
            ("Part", "P-9010"),
            ("Part", "P-9042"),
            ("Component", "компрессор"),
            ("Component", "пусковое реле"),
            ("Supplier", "КомпрессорМаш"),
            ("Supplier", "ХолодКомплект"),
        ],
        [
            ("Model", "AR-420", "BELONGS_TO", "ProductLine", "Арктика"),
            ("Model", "AR-420", "HAS_PART", "Part", "P-9010"),
            ("Part", "P-9010", "IS_A", "Component", "компрессор"),
            ("Part", "P-9010", "SUPPLIED_BY", "Supplier", "КомпрессорМаш"),
            ("Model", "AR-420", "HAS_PART", "Part", "P-9042"),
            ("Part", "P-9042", "IS_A", "Component", "пусковое реле"),
            ("Part", "P-9042", "SUPPLIED_BY", "Supplier", "ХолодКомплект"),
        ],
    ),
)

RETRY_NOTE = (
    "Твой предыдущий ответ не разобрался по схеме. Верни строго JSON-объект вида "
    '{"entities": [{"type": "...", "name": "..."}], '
    '"relations": [{"subject_type": "...", "subject": "...", "type": "...", "object_type": "...", "object": "..."}]}, '
    "без пояснений вокруг него."
)


def extract_chunk(chunk: Chunk, llm: LLMClient) -> ExtractionResult:
    """Возвращает сущности и связи чанка строго по схеме графа.

    Модель вызывается через `llm.chat(..., json_schema=EXTRACTION_SCHEMA)`. Запрос описывает схему
    с направлением каждой связи и содержит хотя бы один пример. Всё, что не подтверждается текстом,
    отбрасывается и считается в `dropped`: код, которого нет в чанке, название не из словаря,
    связь с отброшенным концом. Пригодятся `find_codes` и `canonical_name` из `assistant.infra.grounding`.
    Функция не падает ни на каком ответе модели: одна повторная попытка, затем пустой результат.
    """

    messages = _build_messages(chunk)
    answer = llm.chat(messages, json_schema=EXTRACTION_SCHEMA, max_tokens=2048)
    data = _parse(answer.content)

    if data is None:
        retry = [*messages, {"role": "assistant", "content": answer.content}, {"role": "user", "content": RETRY_NOTE}]
        answer = llm.chat(retry, json_schema=EXTRACTION_SCHEMA, max_tokens=2048)
        data = _parse(answer.content)

    if data is None:
        return ExtractionResult(chunk.chunk_id, chunk.doc_id)

    return _ground(chunk, data)


def _build_messages(chunk: Chunk) -> list[dict]:
    schema = "\n".join(
        f"- {src} -> {dst}: {kind} ({RELATION_HINTS[kind]})" for kind, (src, dst) in RELATION_SIGNATURES.items()
    )
    vocab = "\n".join(f"{kind}: {', '.join(names)}" for kind, names in vocabulary().items())
    examples = "\n\n".join(_example(text, entities, relations) for text, entities, relations in EXAMPLES)
    user = (
        "Извлеки из фрагмента документа сущности и связи по схеме графа знаний.\n\n"
        f"Типы связей и их направление (связь ведёт от левой сущности к правой):\n{schema}\n\n"
        f"{RULES}\n\n"
        f"Словарь названий:\n{vocab}\n\n"
        "Примеры:\n"
        f"{examples}\n\n"
        f"Фрагмент:\n{chunk.text}\n\n"
        "Ответ (только JSON):"
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def _example(text: str, entities: list, relations: list) -> str:
    answer = {
        "entities": [{"type": kind, "name": name} for kind, name in entities],
        "relations": [
            {"subject_type": subject_type, "subject": subject, "type": rel, "object_type": object_type, "object": obj}
            for subject_type, subject, rel, object_type, obj in relations
        ],
    }
    return f"Фрагмент:\n{text}\n\nОтвет:\n{json.dumps(answer, ensure_ascii=False, indent=1)}"


def _parse(content: str) -> tuple[list, list] | None:
    text = content.strip()

    if text.startswith("```"):
        text = text[3:].lstrip()

        if text[:4].lower() == "json":
            text = text[4:]

        text = text.strip()

        if text.endswith("```"):
            text = text[: text.rfind("```")].strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None

    entities, relations = data.get("entities"), data.get("relations")

    if not isinstance(entities, list) or not isinstance(relations, list):
        return None

    if not all(_validate_entity(item) for item in entities) or not all(_validate_relation(item) for item in relations):
        return None

    return entities, relations


def _validate_entity(item: object) -> bool:
    return (
        isinstance(item, dict)
        and item.get("type") in ENTITY_TYPES
        and isinstance(item.get("name"), str)
    )


def _validate_relation(item: object) -> bool:
    return (
        isinstance(item, dict)
        and item.get("subject_type") in ENTITY_TYPES
        and isinstance(item.get("subject"), str)
        and item.get("type") in RELATION_SIGNATURES
        and item.get("object_type") in ENTITY_TYPES
        and isinstance(item.get("object"), str)
    )


def _ground(chunk: Chunk, data: tuple[list, list]) -> ExtractionResult:
    vocab = vocabulary()
    codes = set(find_codes(chunk.text))
    names = set(find_names(chunk.text, vocab))
    entities = {}
    relations = []
    dropped = 0

    for item in data[0]:
        entity = _confirm(item["type"], item["name"], codes, names, vocab)

        if entity:
            entities[entity.node_id] = entity
        else:
            dropped += 1

    for item in data[1]:
        subject = _confirm(item["subject_type"], item["subject"], codes, names, vocab)
        obj = _confirm(item["object_type"], item["object"], codes, names, vocab)
        relation = Relation(subject, item["type"], obj) if subject and obj else None

        if relation and not relation.matches_schema():
            relation = None

        if relation:
            relations.append(relation)
        else:
            dropped += 1

        for end in (subject, obj):
            if end:
                entities[end.node_id] = end

    return ExtractionResult(chunk.chunk_id, chunk.doc_id, list(entities.values()), relations, dropped)


def _confirm(
    entity_type: str, raw: str, codes: set[str], names: set[tuple[str, str]], vocab: dict[str, list[str]]
) -> Entity | None:
    name = canonical_name(entity_type, raw, vocab)

    if name is None:
        return None

    confirmed = name in codes if entity_type in CODE_TYPES else (entity_type, name) in names

    return Entity(entity_type, name) if confirmed else None
