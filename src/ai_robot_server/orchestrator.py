from __future__ import annotations

from typing import Any

from .connectors import RagflowClient, XinferenceClient


class ConversationOrchestrator:
    """Coordinates model and knowledge-base calls for management test queries."""

    def __init__(self, ragflow: RagflowClient, xinference: XinferenceClient) -> None:
        self.ragflow = ragflow
        self.xinference = xinference

    async def query(self, question: str, context: dict[str, Any]) -> dict[str, Any]:
        if self.ragflow.configured:
            return await self.ragflow.query(question, context)
        models = await self.xinference.list_models()
        return {
            "answer": "",
            "sources": [],
            "models": models,
            "message": "ragflow is not configured; query was not sent",
        }
