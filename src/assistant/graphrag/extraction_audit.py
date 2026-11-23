"""Разбор ошибок извлечения: python -m assistant.graphrag.extraction_audit.

Эталон используется только для оценки результата, а не при извлечении или сборке графа.
Повторные запросы используют кеш LLMClient. Инфраструктурная оценка остаётся неизменной.
"""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from assistant.graphrag.extraction import _ExtractionRecord, extract_chunk
from assistant.infra.chunking import load_chunks
from assistant.infra.config import Settings, make_llm, vocabulary
from assistant.infra.evaluation import evaluate_extraction
from assistant.infra.grounding import CODE_TYPES, canonical_name, find_codes
from assistant.infra.schemas import Chunk, Entity, Relation


class _RecordedLLM:
    """Сохраняет последний сырой ответ для отдельного подсчёта отброшенных связей."""

    def __init__(self, client):
        self.client = client
        self.content = ""

    def chat(self, *args, **kwargs):
        response = self.client.chat(*args, **kwargs)
        self.content = response.content
        return response


def _dropped_relations(chunk: Chunk, content: str) -> int | None:
    try:
        record = _ExtractionRecord.model_validate_json(content)
    except ValidationError:
        return None  # пустой результат после двух неразбираемых ответов
    names = vocabulary()
    codes = set(find_codes(chunk.text))

    def endpoint(kind: str, raw: str) -> Entity | None:
        name = canonical_name(kind, raw, names)
        if name is None or (kind in CODE_TYPES and name not in codes):
            return None
        return Entity(kind, name)

    dropped = 0
    for relation in record.relations:
        subject = endpoint(relation.subject_type, relation.subject)
        target = endpoint(relation.object_type, relation.object)
        if subject is None or target is None or not Relation(subject, relation.type, target).matches_schema():
            dropped += 1
    return dropped


def main() -> None:
    settings = Settings.from_env()
    llm = make_llm(settings)
    recorded = _RecordedLLM(llm)
    gold = json.loads((settings.data_dir / "gold_dev.json").read_text(encoding="utf-8"))
    results = []
    dropped_relations = 0
    invalid_chunks = []
    for chunk in load_chunks(settings.data_dir / "corpus"):
        if chunk.doc_id in gold:
            results.append(extract_chunk(chunk, recorded))
            count = _dropped_relations(chunk, recorded.content)
            if count is None:
                invalid_chunks.append(chunk.chunk_id)
            else:
                dropped_relations += count
            print(f"проверен {chunk.chunk_id}", flush=True)
    metrics = evaluate_extraction(gold, results)
    documents = []
    for doc_id, expected in gold.items():
        selected = [result for result in results if result.doc_id == doc_id]
        found = {
            (relation.subject.node_id, relation.type, relation.object.node_id)
            for result in selected for relation in result.relations
        }
        required = {tuple(relation) for relation in expected["relations"]}
        allowed = required | {tuple(relation) for relation in expected["optional_relations"]}
        documents.append({
            "doc_id": doc_id,
            "chunks": len(selected),
            "found_relations": len(found),
            "required_relations": len(required),
            "missing_relations": sorted(required - found),
            "extra_relations": sorted(found - allowed),
            "dropped_entities_and_relations": sum(result.dropped for result in selected),
        })
    report = {
        "model": llm.model,
        "extraction_source_sha256": hashlib.sha256(Path(__file__).with_name("extraction.py").read_bytes()).hexdigest(),
        "created_at": datetime.now(UTC).isoformat(),
        "metrics": metrics,
        "dropped_relations": dropped_relations,
        "invalid_response_chunks": invalid_chunks,
        "llm": llm.stats.as_dict(),
        "documents": documents,
        "note": "dropped — сумма отброшенных сущностей и связей, как в ExtractionResult; "
                "extra_relations — связи вне обязательного и допустимого эталона.",
    }
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    path = settings.artifacts_dir / "extraction_analysis.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False), flush=True)
    print(f"разбор сохранён: {path}", flush=True)


if __name__ == "__main__":
    main()
