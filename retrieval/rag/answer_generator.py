# code/retrieval/rag/answer_generator.py
from __future__ import annotations

import logging
import re
from typing import Optional

from langchain_ollama import ChatOllama
from code.retrieval.rag.config import LOCAL_LLM_MODEL_ANSWER, LLM_BASE_URL

logger = logging.getLogger(__name__)


def _get_llm(llm: Optional[ChatOllama] = None) -> ChatOllama:
    if llm is not None:
        return llm
    return ChatOllama(model=LOCAL_LLM_MODEL_ANSWER, temperature=0.0, base_url=LLM_BASE_URL)


_BASE = """Ты корпоративный ассистент. Отвечаешь строго по контексту из нормативных документов.

Правила:
- Используй ТОЛЬКО информацию из контекста. Никаких внешних знаний.
- Ничего не придумывай и не додумывай.
- Если есть конкретное число, срок, периодичность — воспроизведи точно, процитируй дословно.
- Не переформулируй факты в противоположный смысл.
- Терминология в документе может отличаться от вопроса — ищи по смыслу."""

_PROMPTS = {

    "norm_reference": """{base}

Вопрос про то где прописано правило, требование или процедура.
Формат ответа:
Документ: [название]
Раздел: [номер и название]
Суть: [одно предложение — что там написано]

Выбери один наиболее релевантный источник.

КОНТЕКСТ:
{context}

ВОПРОС: {question}
ОТВЕТ:""",

    "procedure": """{base}

Вопрос про порядок действий или процедуру.
- Если есть явные шаги — выведи их нумерованным списком.
- Если шагов нет но есть требования — выведи списком.
- Сохраняй нумерацию из документа.

КОНТЕКСТ:
{context}

ВОПРОС: {question}
ОТВЕТ:""",

    "definition": """{base}

Вопрос про определение термина или понятия.
Формат: «Термин — определение из документа»
Если есть уточняющие элементы — добавь кратко.

КОНТЕКСТ:
{context}

ВОПРОС: {question}
ОТВЕТ:""",

    "fact_single": """{base}

Вопрос про конкретный факт, требование или да/нет.

Правила:
- Дай прямой короткий ответ — одно-два предложения.
- Строго следуй тому что написано в контексте — не меняй смысл на противоположный.
- Если контекст говорит что что-то ДОПУСКАЕТСЯ или МОЖНО ПРОПУСТИТЬ — отвечай что да, допускается.
- Если контекст говорит что что-то ОБЯЗАТЕЛЬНО — отвечай что да, обязательно.
- Процитируй ключевую фразу из документа дословно.
- Не добавляй оговорок если их нет в контексте.

КОНТЕКСТ:
{context}

ВОПРОС: {question}
ОТВЕТ:""",

    "list": """{base}

Вопрос предполагает список.
- Выведи все пункты списком, не сворачивай несколько в один.
- Если есть явная нумерация в документе — сохрани её.
- Если списка нет явно — извлеки пункты из текста.

КОНТЕКСТ:
{context}

ВОПРОС: {question}
ОТВЕТ:""",

    "broad_overview": """{base}

Вопрос обзорный — про структуру, состав, функции или общую информацию.
- Ответь используя всё релевантное из контекста.
- Если есть пункты или подпункты — выведи списком.
- Не пересказывай всё — только то что относится к вопросу.

КОНТЕКСТ:
{context}

ВОПРОС: {question}
ОТВЕТ:""",
}


def _invoke(prompt: str, llm: ChatOllama) -> str:
    response = llm.invoke([
        {"role": "system", "content": "Отвечай ТОЛЬКО на русском языке."},
        {"role": "user",   "content": prompt},
    ])
    raw = response.content
    logger.info(f"[LLM] сырой ответ ({len(raw)} символов): {raw[:500]!r}{'...' if len(raw) > 500 else ''}")
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    if not cleaned:
        logger.warning("[LLM] после удаления <think> ответ пустой, возвращаем сырой")
        cleaned = raw.strip()
    logger.info(f"[LLM] финальный ответ ({len(cleaned)} символов): {cleaned[:300]!r}{'...' if len(cleaned) > 300 else ''}")
    return cleaned


def _select_template(question_type: str, answer_mode: str) -> str:
    if question_type == "norm_reference":
        return "norm_reference"
    if question_type == "procedure" or answer_mode == "steps":
        return "procedure"
    if question_type == "definition":
        return "definition"
    if question_type == "broad_overview":
        return "broad_overview"
    if answer_mode == "list":
        return "list"
    return "fact_single"


def generate_answer(
    question: str,
    context: str,
    question_type: str,
    answer_mode: str,
    llm: Optional[ChatOllama] = None,
) -> str:
    _llm = _get_llm(llm)
    template_key = _select_template(question_type, answer_mode)
    template = _PROMPTS[template_key]
    prompt = template.format(base=_BASE, context=context, question=question)
    logger.info(f"[ANSWER] template={template_key}  context={len(context)} символов")
    return _invoke(prompt, _llm)