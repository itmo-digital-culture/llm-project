"""Инструменты агента. Требование FR-6."""

from collections.abc import Callable

from assistant.graphrag.search import global_search, local_search
from assistant.infra.erp_client import ErpClient, ErpNotFound, ErpRejected
from assistant.infra.grounding import code_type, normalize_code
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import Answer, KnowledgeBase

WAREHOUSES = ["Москва", "Санкт-Петербург", "Казань", "Новосибирск"]


def _function_schema(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


TOOL_SCHEMAS: list[dict] = [
    _function_schema(
        "search_knowledge",
        "Ищет конкретный факт в графе знаний: ошибка, модель, деталь, замена, поставщик, регламент. "
        "Вызывай, когда номера детали ещё нет в вопросе.",
        {"question": {"type": "string", "description": "Вопрос к базе знаний, без выдуманных номеров деталей"}},
        ["question"],
    ),
    _function_schema(
        "summarize_topic",
        "Обзор по всему корпусу: частые отказы линейки, модели с наибольшим числом отказов, "
        "детали, снятые с производства.",
        {"question": {"type": "string", "description": "Обзорный вопрос"}},
        ["question"],
    ),
    _function_schema(
        "check_stock",
        "Остаток детали на складе и признак discontinued. Остаток 0 значит, что деталь известна, но закончилась.",
        {
            "part_id": {"type": "string", "description": "Номер детали вида P-1042 или P-1042B"},
            "warehouse": {"type": "string", "enum": WAREHOUSES, "description": "Склад"},
        },
        ["part_id", "warehouse"],
    ),
    _function_schema(
        "get_price",
        "Цена детали в рублях и признак discontinued.",
        {"part_id": {"type": "string", "description": "Номер детали вида P-1042 или P-1042B"}},
        ["part_id"],
    ),
    _function_schema(
        "create_service_order",
        "Создаёт заказ детали. Количество должно быть названо пользователем. "
        "Если деталь заменена, заказывай последнюю в цепочке замены.",
        {
            "part_id": {"type": "string", "description": "Номер детали вида P-1042 или P-1042B"},
            "warehouse": {"type": "string", "enum": WAREHOUSES, "description": "Склад"},
            "qty": {"type": "integer", "minimum": 1, "description": "Число штук, названное пользователем"},
            "model": {"type": "string", "description": "Код модели, например AR-210"},
        },
        ["part_id", "warehouse", "qty", "model"],
    ),
]


def _part_id(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    code = normalize_code(value)
    if code is None or code_type(code) != "Part":
        return None
    return code


def _positive_qty(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, float) and value.is_integer() and value > 0:
        return int(value)
    return None


def _warehouse(value: object) -> str | None:
    if isinstance(value, str) and value in WAREHOUSES:
        return value
    return None


def _erp(action: Callable[[], dict]) -> dict:
    try:
        return action()
    except (ErpNotFound, ErpRejected) as exc:
        return {"error": str(exc)}


def _search_payload(answer: Answer, *, with_facts: bool) -> dict:
    payload = {"answer": answer.text, "sources": answer.sources, "refused": answer.refused}
    if with_facts:
        payload["facts"] = answer.facts
    return payload


def build_tools(kb: KnowledgeBase, llm: LLMClient, erp: ErpClient) -> dict[str, Callable[..., dict]]:
    """Словарь «имя инструмента -> функция». Каждая функция возвращает словарь.

    Функция проверяет свои аргументы до обращения к ERP: склад из списка, номер детали вида P-1042
    или P-1042B, количество больше нуля. При ошибке возвращает {"error": "..."} и ничего не выполняет.
    `ErpNotFound` и `ErpRejected` превращаются в {"error": "..."} с разным текстом, чтобы исходы
    оставались различимыми. `ErpUnavailable` передаётся дальше: повторы делает агент (FR-8).
    """

    def search_knowledge(question: str) -> dict:
        return _search_payload(local_search(question, kb, llm), with_facts=True)

    def summarize_topic(question: str) -> dict:
        return _search_payload(global_search(question, kb, llm), with_facts=False)

    def check_stock(part_id: str, warehouse: str) -> dict:
        code = _part_id(part_id)
        if code is None:
            return {"error": "номер детали должен быть вида P-1042 или P-1042B"}
        place = _warehouse(warehouse)
        if place is None:
            return {"error": f"неизвестный склад: {warehouse}"}
        return _erp(lambda: erp.get_stock(code, place))

    def get_price(part_id: str) -> dict:
        code = _part_id(part_id)
        if code is None:
            return {"error": "номер детали должен быть вида P-1042 или P-1042B"}
        return _erp(lambda: erp.get_price(code))

    def create_service_order(part_id: str, warehouse: str, qty: int, model: str) -> dict:
        code = _part_id(part_id)
        if code is None:
            return {"error": "номер детали должен быть вида P-1042 или P-1042B"}
        place = _warehouse(warehouse)
        if place is None:
            return {"error": f"неизвестный склад: {warehouse}"}
        amount = _positive_qty(qty)
        if amount is None:
            return {"error": "количество должно быть целым числом больше нуля"}
        if not isinstance(model, str) or not model.strip():
            return {"error": "не указана модель"}
        return _erp(lambda: erp.create_order(code, place, amount, model.strip()))

    return {
        "search_knowledge": search_knowledge,
        "summarize_topic": summarize_topic,
        "check_stock": check_stock,
        "get_price": get_price,
        "create_service_order": create_service_order,
    }
