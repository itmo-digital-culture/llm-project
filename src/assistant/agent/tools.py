"""Инструменты агента. Требование FR-6."""

from collections.abc import Callable

from assistant.infra.erp_client import ErpClient
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import KnowledgeBase

# Описания пяти инструментов в формате function calling:
# search_knowledge, summarize_topic, check_stock, get_price, create_service_order.
# TODO: заполните список. У параметра warehouse четыре склада задаются перечислением.
TOOL_SCHEMAS: list[dict] = []


def build_tools(kb: KnowledgeBase, llm: LLMClient, erp: ErpClient) -> dict[str, Callable[..., dict]]:
    """Словарь «имя инструмента -> функция». Каждая функция возвращает словарь.

    Функция проверяет свои аргументы до обращения к ERP: склад из списка, номер детали вида P-1042
    или P-1042B, количество больше нуля. При ошибке возвращает {"error": "..."} и ничего не выполняет.
    Исключения клиента ERP (`ErpNotFound`, `ErpRejected`, `ErpUnavailable`) означают разные исходы,
    и в результате инструмента они должны остаться различимыми.
    """
    # TODO: пять функций поверх local_search, global_search и клиента ERP
    raise NotImplementedError("Реализуйте build_tools")
