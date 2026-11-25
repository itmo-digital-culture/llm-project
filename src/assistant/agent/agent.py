"""Цикл агента. Требования FR-7, FR-8, FR-9 и FR-10."""

import json
import re
from collections.abc import Callable

from assistant.agent.tools import TOOL_SCHEMAS
from assistant.infra.erp_client import ErpUnavailable
from assistant.infra.grounding import code_type, find_codes
from assistant.infra.llm import LLMClient, tool_message
from assistant.infra.schemas import AgentResult, ToolStep

_CODE_SPAN = re.compile(r"(?<!\w)[A-Za-zА-Яа-яЁё]{1,2}[- ]?\d{2,4}[A-Za-zА-Яа-яЁё]?(?!\w)")
_NUMBER = r"(?<![\w-])(?:\d{1,3}(?:[ \u00a0\u202f]\d{3})+|\d+)(?![\w-])"
_NUMBER_RE = re.compile(_NUMBER)
_WITH_UNIT = re.compile(rf"({_NUMBER})\s*(руб|₽|шт)")
_QTY_WORDS = {
    **dict.fromkeys(("один", "одну", "одна", "одного"), "1"),
    **dict.fromkeys(("два", "две", "двух"), "2"),
    **dict.fromkeys(("три", "трех", "трёх"), "3"),
    "четыре": "4",
    "пять": "5",
    "шесть": "6",
    "семь": "7",
    "восемь": "8",
    "девять": "9",
    "десять": "10",
}
_KB_REFUSAL = "в базе знаний не найдено"
_GROUNDING_REFUSAL = "Ответ не подтверждается вопросом и результатами инструментов."
_SYSTEM_PROMPT = """Ты ассистент диспетчера «ПолярТех». Отвечай по-русски и не выдумывай коды, цены и остатки.

search_knowledge - конкретный факт, если номера детали в вопросе ещё нет.
summarize_topic - обзор: частые отказы и снятые с производства детали.
check_stock - остаток. 0 значит «закончилась», ошибка значит «данных нет».
get_price - цена в рублях.
create_service_order - заказ, только если пользователь сам назвал число штук. Бери последнюю деталь цепочки замены.

Цену пиши как «2630 руб», остаток как «3 шт». Если данных нет, напиши: «в базе знаний не найдено».
Без подтверждения человека заказ не создан.
"""


def run_agent(
    question: str,
    llm: LLMClient,
    tools: dict[str, Callable[..., dict]],
    *,
    max_steps: int = 6,
    confirm: bool = False,
) -> AgentResult:
    """Ведёт диалог с моделью в цикле «рассуждение, вызов инструмента, наблюдение».

    Модель получает `TOOL_SCHEMAS` через `llm.chat(..., tools=...)`. Каждый вызов записывается
    в `steps` как `ToolStep`. Агент сам проверяет номер детали в аргументах (он должен встречаться
    в вопросе или в результатах прошлых инструментов), количество в заказе (должно быть названо
    пользователем) и повторные вызовы. При `ErpUnavailable` вызов повторяется до двух раз.
    Заказ создаётся только при `confirm=True`, иначе агент возвращает `needs_confirmation=True`
    и `pending_action`. Перед выдачей ответа коды, цены и остатки сверяются с вопросом
    и результатами инструментов.
    """
    messages = [{"role": "system", "content": _SYSTEM_PROMPT}, {"role": "user", "content": question}]
    steps: list[ToolStep] = []
    sources: list[str] = []
    cache: dict[str, tuple[dict, str | None]] = {}
    needs_confirmation = False
    pending_action: dict | None = None

    def finish(text: str) -> AgentResult:
        answer = _with_order_id(text.strip(), steps)
        problems = _grounding_problems(answer, question, steps)
        if problems:
            messages.append({"role": "user", "content": _repair_prompt(problems)})
            repaired = llm.chat(messages)
            messages.append(repaired.as_message())
            answer = _with_order_id(repaired.content.strip(), steps)
            if not answer or _grounding_problems(answer, question, steps):
                return _result(_GROUNDING_REFUSAL, refused=True)
        if not answer:
            return _result(_KB_REFUSAL, refused=True)
        return _result(answer, refused=_KB_REFUSAL in answer.lower())

    def _result(answer: str, *, refused: bool) -> AgentResult:
        return AgentResult(
            answer,
            steps=list(steps),
            sources=list(sources),
            refused=refused,
            needs_confirmation=needs_confirmation,
            pending_action=pending_action,
        )

    while len(steps) < max_steps:
        response = llm.chat(messages, tools=None if needs_confirmation else TOOL_SCHEMAS)
        messages.append(response.as_message())
        if needs_confirmation or not response.tool_calls:
            return finish(response.content)

        for call in response.tool_calls:
            if len(steps) >= max_steps:
                break
            payload, step, pending = _handle_call(
                call.name,
                dict(call.arguments),
                question,
                steps,
                tools,
                cache,
                confirm=confirm,
            )
            steps.append(step)
            _collect_sources(sources, payload, step.error)
            messages.append(tool_message(call, payload))
            if pending is not None:
                needs_confirmation = True
                pending_action = pending
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Заказ не создан и ждёт подтверждения человека. "
                            "Ответь по уже полученным данным и не вызывай инструменты."
                        ),
                    }
                )
                break

    learned = _learned(steps)
    return _result(f"{learned} Работа прервана: исчерпан лимит шагов.", refused=False)


def _handle_call(
    name: str,
    args: dict,
    question: str,
    steps: list[ToolStep],
    tools: dict[str, Callable[..., dict]],
    cache: dict[str, tuple[dict, str | None]],
    *,
    confirm: bool,
) -> tuple[dict, ToolStep, dict | None]:
    if name not in tools:
        return _rejected(name, args, f"неизвестный инструмент {name}")
    if "_raw" in args:
        return _rejected(name, args, "аргументы инструмента не читаются")

    unknown = [part for part in _argument_parts(args) if part not in _known_parts(question, steps)]
    if unknown:
        listed = ", ".join(unknown)
        return _rejected(
            name,
            args,
            f"Номер детали {listed} не встречается в вопросе и в результатах инструментов. Вызовите search_knowledge.",
        )
    if name == "create_service_order" and not _qty_named(question, args.get("qty")):
        return _rejected(
            name,
            args,
            "Количество в заказе не названо в вопросе. Спросите у пользователя, сколько штук заказать.",
        )
    if name == "create_service_order" and not confirm:
        payload = {"error": "Заказ не создан: нужно подтверждение человека."}
        return payload, ToolStep(name, args, error=payload["error"]), {"tool": name, "args": args}

    try:
        key = json.dumps({"tool": name, "args": args}, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return _rejected(name, args, "аргументы инструмента не читаются")
    if key in cache:
        payload, error = cache[key]
        return payload, ToolStep(name, args, result=None if error else payload, error=error), None

    payload, error = _execute(tools[name], args)
    cache[key] = (payload, error)
    return payload, ToolStep(name, args, result=None if error else payload, error=error), None


def _rejected(name: str, args: dict, error: str) -> tuple[dict, ToolStep, None]:
    payload = {"error": error}
    return payload, ToolStep(name, args, error=error), None


def _execute(fn: Callable[..., dict], args: dict) -> tuple[dict, str | None]:
    error: str | None = None
    for _attempt in range(3):
        try:
            payload = fn(**args)
        except ErpUnavailable as exc:
            error = str(exc) or "ERP недоступна"
            continue
        except Exception as exc:
            error = str(exc) or "инструмент завершился с ошибкой"
            return {"error": error}, error
        if not isinstance(payload, dict):
            error = "инструмент вернул не словарь"
            return {"error": error}, error
        reported = payload.get("error")
        return payload, str(reported) if reported else None
    return {"error": "Данные ERP получить не удалось"}, "Данные ERP получить не удалось"


def _argument_parts(args: dict) -> list[str]:
    try:
        blob = json.dumps(args, ensure_ascii=False)
    except TypeError:
        return []
    return [code for code in find_codes(blob) if code_type(code) == "Part"]


def _known_parts(question: str, steps: list[ToolStep]) -> set[str]:
    chunks = [question]
    chunks.extend(json.dumps(step.result, ensure_ascii=False) for step in steps if step.result)
    return {code for code in find_codes("\n".join(chunks)) if code_type(code) == "Part"}


def _qty_named(question: str, qty: object) -> bool:
    if isinstance(qty, bool):
        return False
    if isinstance(qty, int):
        number = str(qty) if qty > 0 else ""
    elif isinstance(qty, float) and qty.is_integer() and qty > 0:
        number = str(int(qty))
    elif isinstance(qty, str) and qty.strip().isdigit() and int(qty) > 0:
        number = str(int(qty))
    else:
        return False
    named = {re.sub(r"\D", "", match) for match in _NUMBER_RE.findall(_CODE_SPAN.sub(" ", question))}
    named.update(match.group(1) for match in re.finditer(r"(\d+)\s*шт", question, flags=re.IGNORECASE))
    lowered = question.lower()
    named.update(value for word, value in _QTY_WORDS.items() if re.search(rf"(?<!\w){word}(?!\w)", lowered))
    return number in named


def _grounding_problems(answer: str, question: str, steps: list[ToolStep]) -> list[str]:
    chunks = [question]
    chunks.extend(json.dumps(step.result, ensure_ascii=False) for step in steps if step.error is None and step.result)
    allowed = "\n".join(chunks)
    allowed_codes = set(find_codes(allowed))
    allowed_numbers = set(re.findall(r"\d+", allowed))
    problems = [f"код {code}" for code in find_codes(answer) if code not in allowed_codes]
    for raw, unit in _WITH_UNIT.findall(answer):
        digits = re.sub(r"\D", "", raw)
        if digits not in allowed_numbers:
            problems.append(f"число {digits} {unit}")
    return problems


def _repair_prompt(problems: list[str]) -> str:
    found = ", ".join(problems)
    return (
        "В ответе есть данные, которых нет ни в вопросе, ни в результатах инструментов: "
        f"{found}. Напиши ответ заново и оставь только подтверждённые коды, цены и остатки."
    )


def _with_order_id(answer: str, steps: list[ToolStep]) -> str:
    for step in steps:
        if step.tool != "create_service_order" or step.error is not None or not step.result:
            continue
        order_id = step.result.get("order_id")
        if isinstance(order_id, str) and order_id and order_id not in answer:
            suffix = f"Заказ {order_id} создан."
            answer = f"{answer}\n{suffix}" if answer else suffix
    return answer


def _collect_sources(sources: list[str], payload: dict, error: str | None) -> None:
    if error is not None:
        return
    for source in payload.get("sources") or []:
        if isinstance(source, str) and source not in sources:
            sources.append(source)


def _learned(steps: list[ToolStep]) -> str:
    bits = [json.dumps(step.result, ensure_ascii=False) for step in steps if step.error is None and step.result]
    return " ".join(bits) if bits else "Новых данных нет."
