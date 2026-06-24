from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any


NO_KNOWLEDGE_BASE_ANSWER = "知识库中未找到您要的答案，我可以换个问法继续帮你查。"


class AnswerIntent(str, Enum):
    GREETING = "greeting"
    REPEAT = "repeat"
    CLARIFICATION = "clarification"
    PROJECT_QUESTION = "project_question"
    OUT_OF_SCOPE = "out_of_scope"


@dataclass(frozen=True)
class AnswerDecision:
    intent: AnswerIntent
    use_ragflow: bool
    answer: str | None = None


def decide_answer(question: str, context: dict[str, Any] | None = None) -> AnswerDecision:
    normalized = _normalize_question(question)
    context = context or {}

    if _looks_like_repeat(normalized):
        last_answer = _last_assistant_answer(context)
        if last_answer:
            return AnswerDecision(
                intent=AnswerIntent.REPEAT,
                use_ragflow=False,
                answer=f"我再说一遍：{_shorten_spoken_text(last_answer, max_chars=100)}",
            )
        return AnswerDecision(
            intent=AnswerIntent.REPEAT,
            use_ragflow=False,
            answer="我刚才没有可重复的回答，你可以把问题再说一遍。",
        )

    if _looks_like_greeting(normalized):
        return AnswerDecision(
            intent=AnswerIntent.GREETING,
            use_ragflow=False,
            answer="我是机器视觉工作室的语音讲解助手，可以介绍项目、方案、设备和演示流程。",
        )

    if _looks_like_clarification(normalized):
        return AnswerDecision(
            intent=AnswerIntent.CLARIFICATION,
            use_ragflow=False,
            answer="没关系，你可以换个说法，问我项目、方案、设备或流程相关内容。",
        )

    if _looks_obviously_out_of_scope(normalized):
        return AnswerDecision(
            intent=AnswerIntent.OUT_OF_SCOPE,
            use_ragflow=False,
            answer="这个问题超出演示范围，我可以回答项目、方案、设备和流程相关内容。",
        )

    return AnswerDecision(intent=AnswerIntent.PROJECT_QUESTION, use_ragflow=True)


def normalize_ragflow_answer(answer: str, rag_response: dict[str, Any]) -> str:
    cleaned = _clean_spoken_text(answer)
    if _is_knowledge_base_miss(cleaned) or _has_explicit_empty_sources(rag_response):
        return NO_KNOWLEDGE_BASE_ANSWER
    return _shorten_spoken_text(cleaned, max_chars=120)


def _normalize_question(question: str) -> str:
    return re.sub(r"\s+", "", str(question).strip().lower())


def _looks_like_repeat(question: str) -> bool:
    if not question:
        return False
    repeat_markers = (
        "再说一遍",
        "重复一下",
        "重复一遍",
        "重说",
        "刚才没听清",
        "我没听清",
        "没听清",
        "没听到",
        "刚才说什么",
        "上一句",
    )
    return any(marker in question for marker in repeat_markers)


def _looks_like_greeting(question: str) -> bool:
    if not question:
        return False
    greeting_markers = (
        "你好",
        "您好",
        "hello",
        "hi",
        "嗨",
        "你是谁",
        "你叫什么",
        "介绍一下你自己",
        "介绍自己",
    )
    return any(marker in question for marker in greeting_markers) and len(question) <= 18


def _looks_like_clarification(question: str) -> bool:
    if not question or len(question) > 16:
        return False
    clarification_markers = (
        "什么意思",
        "没听懂",
        "听不懂",
        "我不明白",
        "不明白",
        "说清楚点",
    )
    return any(marker in question for marker in clarification_markers)


def _looks_obviously_out_of_scope(question: str) -> bool:
    if not question:
        return False
    out_of_scope_markers = (
        "讲个笑话",
        "说个笑话",
        "唱首歌",
        "今天天气",
        "天气怎么样",
        "几点了",
        "股票",
        "彩票",
        "世界杯",
        "新闻",
        "菜谱",
        "写首诗",
        "翻译成英文",
        "英语怎么说",
    )
    return any(marker in question for marker in out_of_scope_markers)


def _last_assistant_answer(context: dict[str, Any]) -> str:
    history = context.get("conversation_history", [])
    if not isinstance(history, list):
        return ""
    for item in reversed(history):
        if not isinstance(item, dict):
            continue
        if str(item.get("role", "")).strip() != "assistant":
            continue
        content = _clean_spoken_text(str(item.get("content", "")))
        if content:
            return content
    return ""


def _clean_spoken_text(text: str) -> str:
    cleaned = str(text).strip()
    cleaned = re.sub(r"```.*?```", "", cleaned, flags=re.S)
    cleaned = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", cleaned)
    cleaned = re.sub(r"\[[^\]]+\]\([^)]+\)", "", cleaned)
    cleaned = re.sub(r"[*_#>`-]+", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _shorten_spoken_text(text: str, *, max_chars: int) -> str:
    cleaned = _clean_spoken_text(text)
    if len(cleaned) <= max_chars:
        return cleaned

    candidate = cleaned[:max_chars]
    for punct in "。！？；!?;":
        index = candidate.rfind(punct)
        if index >= 40:
            return candidate[: index + 1]
    return candidate.rstrip("，,：:；;、 ") + "。"


def _is_knowledge_base_miss(answer: str) -> bool:
    compact = _normalize_question(answer)
    return "知识库中未找到您要的答案" in compact or "知识库未找到" in compact


def _has_explicit_empty_sources(rag_response: dict[str, Any]) -> bool:
    source_values = list(_iter_source_values(rag_response))
    if not source_values:
        return False
    return all(_is_empty_source_value(value) for value in source_values)


def _iter_source_values(value: Any):
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"sources", "source", "references", "reference", "rag_sources"}:
                yield child
            yield from _iter_source_values(child)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_source_values(item)


def _is_empty_source_value(value: Any) -> bool:
    if value is None:
        return True
    if value == "":
        return True
    if isinstance(value, (list, tuple, set)):
        return len(value) == 0
    if isinstance(value, dict):
        if not value:
            return True
        for key in ("chunks", "docs", "documents", "items"):
            if key in value and not value[key]:
                return True
    return False
