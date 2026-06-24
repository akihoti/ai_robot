from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
import logging
from typing import Any

from .answer_policy import decide_answer, normalize_ragflow_answer
from .connectors import MeloTtsClient, RagflowClient, XinferenceClient
from .config import TtsConfig, VoiceGatewayConfig


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class VoiceGatewayResult:
    question: str
    answer: str
    audio_bytes: bytes
    media_type: str
    rag_response: dict[str, Any]


class ConversationOrchestrator:
    """Coordinates management queries and the voice gateway pipeline."""

    def __init__(
        self,
        ragflow: RagflowClient,
        xinference: XinferenceClient,
        voice_gateway: VoiceGatewayConfig,
        *,
        tts_client: MeloTtsClient | None = None,
        tts_config: TtsConfig | None = None,
    ) -> None:
        self.ragflow = ragflow
        self.xinference = xinference
        self.voice_gateway = voice_gateway
        self.tts_client = tts_client
        self.tts_config = tts_config or TtsConfig()

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

    async def text_chat(self, question: str, context: dict[str, Any]) -> dict[str, Any]:
        answer, rag_response = await self.answer_question(question, context)
        return {
            "question": question,
            "answer": answer,
            "rag_response": rag_response,
        }

    async def voice_chat(
        self,
        *,
        filename: str,
        audio_bytes: bytes,
        content_type: str,
        context: dict[str, Any],
    ) -> VoiceGatewayResult:
        self._validate_voice_gateway(require_ragflow=False)
        question = await self.transcribe_audio(
            filename=filename,
            audio_bytes=audio_bytes,
            content_type=content_type,
        )
        answer, rag_response = await self.answer_question(question, context)
        audio_out, media_type = await self.synthesize_text(answer)
        return VoiceGatewayResult(
            question=question,
            answer=answer,
            audio_bytes=audio_out,
            media_type=media_type or self.voice_gateway.tts_media_type,
            rag_response=rag_response,
        )

    async def transcribe_audio(
        self,
        *,
        filename: str,
        audio_bytes: bytes,
        content_type: str,
    ) -> str:
        self._validate_voice_gateway(require_ragflow=False, require_tts=False)
        asr_response = await self.xinference.transcribe_audio(
            model=self.voice_gateway.asr_model,
            filename=filename,
            audio_bytes=audio_bytes,
            content_type=content_type,
        )
        return _extract_asr_text(asr_response)

    async def answer_question(
        self,
        question: str,
        context: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        decision = decide_answer(question, context)
        if not decision.use_ragflow:
            answer = decision.answer or ""
            return answer, _policy_response(
                answer=answer,
                intent=decision.intent.value,
                source="local_template",
            )

        self._validate_voice_gateway(require_xinference=False, require_tts=False)
        payload = dict(context or {})
        payload.setdefault("system_prompt", self.voice_gateway.system_prompt)
        rag_response = await self.ragflow.query_chat(
            chat_id=self.voice_gateway.ragflow_chat_id,
            question=question,
            payload=payload,
        )
        raw_answer = _extract_ragflow_answer(rag_response)
        answer = normalize_ragflow_answer(raw_answer, rag_response)
        return answer, _with_policy_metadata(
            rag_response,
            intent=decision.intent.value,
            source="ragflow",
            normalized=answer != raw_answer,
        )

    async def stream_answer(
        self,
        question: str,
        context: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        decision = decide_answer(question, context)
        if not decision.use_ragflow:
            yield decision.answer or ""
            return

        self._validate_voice_gateway(require_xinference=False, require_tts=False)
        payload = dict(context or {})
        payload.setdefault("system_prompt", self.voice_gateway.system_prompt)
        try:
            async for delta in self.ragflow.stream_query_chat(
                chat_id=self.voice_gateway.ragflow_chat_id,
                question=question,
                payload=payload,
            ):
                yield delta
        except Exception:
            answer, _ = await self.answer_question(question, context)
            yield answer

    async def synthesize_text(self, text: str) -> tuple[bytes, str]:
        provider = self.tts_config.provider.strip().lower() or "xinference"
        fallback_provider = self.tts_config.fallback_provider.strip().lower()
        self._validate_voice_gateway(
            require_ragflow=False,
            require_xinference=provider == "xinference" or fallback_provider == "xinference",
            require_asr=False,
        )
        if provider == "melotts_sidecar":
            try:
                if self.tts_client is None or not self.tts_client.configured:
                    raise ValueError("melotts sidecar is not configured")
                return await self.tts_client.synthesize_speech(
                    text=text,
                    voice=self.tts_config.voice or self.voice_gateway.tts_voice,
                    speed=self.tts_config.speed,
                    preferred_media_type=self.voice_gateway.tts_media_type,
                )
            except Exception as exc:
                if fallback_provider != "xinference":
                    raise
                LOGGER.warning(
                    "MeloTTS synthesis failed; falling back to Xinference tts_provider=melotts_sidecar fallback=xinference error=%s",
                    exc,
                )
        return await self.xinference.synthesize_speech(
            model=self.voice_gateway.tts_model,
            text=text,
            voice=self.voice_gateway.tts_voice,
            preferred_media_type=self.voice_gateway.tts_media_type,
        )

    async def build_welcome_audio(self, text: str | None = None) -> tuple[str, bytes, str]:
        welcome_text = (text or self.voice_gateway.welcome_text).strip()
        if not welcome_text:
            raise ValueError("welcome text is empty")
        audio_out, media_type = await self.synthesize_text(welcome_text)
        return welcome_text, audio_out, media_type or self.voice_gateway.tts_media_type

    def _validate_voice_gateway(
        self,
        *,
        require_ragflow: bool = True,
        require_xinference: bool = True,
        require_asr: bool = True,
        require_tts: bool = True,
    ) -> None:
        if require_ragflow and not self.ragflow.configured:
            raise ValueError("ragflow is not configured")
        if require_xinference and not self.xinference.configured:
            raise ValueError("xinference is not configured")
        if require_asr and not self.voice_gateway.asr_model:
            raise ValueError("voice_gateway.asr_model is not configured")
        if require_tts and not self.voice_gateway.tts_model:
            raise ValueError("voice_gateway.tts_model is not configured")
        if require_ragflow and not self.voice_gateway.ragflow_chat_id:
            raise ValueError("voice_gateway.ragflow_chat_id is not configured")


def _extract_asr_text(asr_response: dict[str, Any]) -> str:
    question = str(
        asr_response.get("text") or asr_response.get("data") or ""
    ).strip()
    if not question:
        raise ValueError(f"ASR returned empty text: {asr_response}")
    return question


def _extract_ragflow_answer(rag_response: dict[str, Any]) -> str:
    try:
        answer = str(rag_response["choices"][0]["message"]["content"]).strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"Unexpected RAGFlow response: {rag_response}") from exc
    if not answer:
        raise ValueError("RAGFlow returned empty answer")
    return answer


def _policy_response(*, answer: str, intent: str, source: str) -> dict[str, Any]:
    return {
        "choices": [{"message": {"content": answer}}],
        "answer_policy": {"intent": intent, "source": source},
    }


def _with_policy_metadata(
    rag_response: dict[str, Any],
    *,
    intent: str,
    source: str,
    normalized: bool,
) -> dict[str, Any]:
    enriched = dict(rag_response)
    enriched["answer_policy"] = {
        "intent": intent,
        "source": source,
        "normalized": normalized,
    }
    return enriched
