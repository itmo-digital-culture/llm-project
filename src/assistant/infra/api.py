"""HTTP API ассистента: POST /ask. Запуск: uvicorn assistant.infra.api:app"""

from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel

from assistant.infra.cli import make_context, runner


class AskIn(BaseModel):
    question: str
    mode: Literal["baseline", "local", "global", "agent"] = "agent"
    confirm: bool = False


def create_app() -> FastAPI:
    app = FastAPI(title="Ассистент сервисной службы «ПолярТех»")
    state: dict = {}

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/ask")
    def ask(body: AskIn) -> dict:
        if "context" not in state:
            state["context"] = make_context()
        outcome = runner(body.mode, state["context"], body.confirm)(body.question)
        return {
            "answer": outcome.text,
            "sources": outcome.sources,
            "refused": outcome.refused,
            "needs_confirmation": outcome.needs_confirmation,
            "pending_action": outcome.pending_action,
            "steps": [step.__dict__ for step in outcome.steps],
        }

    return app


app = create_app()
