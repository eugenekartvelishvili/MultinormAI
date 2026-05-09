# code/retrieval/rag/classifier.py
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional

from langchain_ollama import ChatOllama
from code.retrieval.rag.config import LOCAL_LLM_MODEL, LLM_BASE_URL

logger = logging.getLogger(__name__)


def _get_llm(llm: Optional[ChatOllama] = None) -> ChatOllama:
    if llm is not None:
        return llm
    return ChatOllama(model=LOCAL_LLM_MODEL, temperature=0.0, base_url=LLM_BASE_URL)


def _heuristic_answer_mode(question: str) -> str | None:
    q = question.lower().strip()
    if any(x in q for x in [
        "какие требования", "перечень требований", "какие обязанности",
        "какие задачи", "какие функции", "какие правила", "какие условия",
        "какие документы", "какой список", "перечень",
    ]):
        return "list"
    if any(x in q for x in [
        "как ", "каким образом", "порядок", "что делать",
        "как проходит", "как оформить", "какие шаги", "от начала до конца",
    ]):
        return "steps"
    if any(x in q for x in [
        "где прописана", "где указана", "в каком документе",
        "на основании какого", "каким документом", "какой пункт",
        "какой раздел", "где описана", "где указан",
    ]):
        return "reference"
    return None


_PROMPT = """Ты классификатор запросов для корпоративной системы поиска по нормативным документам.

Вопрос: "{question}"

Определи:

1. question_type:
- fact          — точный факт, параметр, числовое требование, да/нет вопрос
    К fact также относятся:
    - "Можно ли", "обязательно ли", "допускается ли", "нужно ли", "надо ли"
    - "кто утверждает", "кто подписывает", "кто отвечает", "кто назначает", "кому", "кто"
    - вопросы про содержимое папок, дисков, каталогов ("что лежит", "какая структура каталогов")
    - перечни документов ("какие документы нужно получить", "что предоставить")
- definition    — определение термина или понятия ("что такое", "что значит", "что называется")
- procedure     — последовательность действий (несколько шагов, порядок выполнения)
    К procedure также относятся: "как оценить", "как транспортировать", "как оформить"
- norm_reference — вопрос про существование или местонахождение документа/регламента/инструкции
    ТОЛЬКО: "есть ли у нас", "есть ли инструкция/положение/регламент", "в каком документе", "где прописано"
    НЕ путать с fact: "есть ли ограничения", "есть ли требования" — это fact
- broad_overview — обзор темы, функций, задач, правил, структуры подразделения, перечень стадий/этапов
    К broad_overview также относятся: "какие стадии", "какие этапы есть в системе", "какие задачи у отдела"

2. needs_decomposition — нужно ли разбить на entity (о чём/о ком) и attribute (что именно):
- true  — вопрос содержит конкретную сущность И атрибут
- false — определение, числовой факт, или вопрос без выраженной сущности

Примеры:
- "Какие задачи у отдела геологии"              → broad_overview, true
- "Как транспортировать отходы IV класса"        → procedure, true
- "Какая структура у ГООС"                      → broad_overview, true
- "Что делать после того как нашли тендер"      → procedure, true
- "Как оценить влияние экологических аспектов"  → procedure, true
- "Что такое финансовые риски"                  → definition, false
- "Надо ли переоценивать поставщиков каждый год" → fact, false
- "Какие требования к оперативной памяти"       → fact, false
- "Где прописана процедура ПТП"                 → norm_reference, false
- "Есть ли у нас положение о командировках"     → norm_reference, false
- "Есть ли у нас инструкция по X"               → norm_reference, false
- "Есть ли рабочая инструкция по структуре каталогов" → norm_reference, false
- "Какие бывают виды рисков"                    → broad_overview, false
- "Какие у нас стадии выполнения работ"         → broad_overview, false
- "Кто утверждает структуру полевой партии"     → fact, false
- "Кто несёт ответственность за учёт отходов"   → fact, false
- "Кто подписывает договор"                     → fact, false
- "Можно ли закупить без оценки поставщика"     → fact, false
- "Сколько можно хранить отходы"                → fact, false
- "Что лежит на диске P"                        → fact, false
- "Какая структура у каталогов диска P"         → fact, false
- "Есть ли ограничения по сроку хранения"       → fact, false
- "Есть ли ограничения по классу опасности"     → fact, false

Верни ТОЛЬКО JSON:
{{
  "question_type": "fact|definition|procedure|norm_reference|broad_overview",
  "needs_decomposition": true|false,
  "confidence": 0.85,
  "reason": "одна фраза"
}}"""


def _extract_json(text: str) -> Dict[str, Any]:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    raise ValueError(f"Не удалось извлечь JSON:\n{text}")


def classify_question(question: str, llm: Optional[ChatOllama] = None) -> Dict[str, Any]:
    _llm = _get_llm(llm)
    prompt = _PROMPT.format(question=question)
    response = _llm.invoke([{"role": "user", "content": prompt}])
    data = _extract_json(response.content)

    allowed_types = {"fact", "definition", "procedure", "norm_reference", "broad_overview"}
    qtype = str(data.get("question_type", "")).strip()
    if qtype not in allowed_types:
        logger.warning(f"[CLASSIFY] неизвестный тип {qtype!r}, fallback → fact")
        qtype = "fact"

    needs_decomposition = bool(data.get("needs_decomposition", False))

    answer_mode = _heuristic_answer_mode(question)
    if answer_mode is None:
        answer_mode = {
            "fact":           "single",
            "definition":     "single",
            "procedure":      "steps",
            "norm_reference": "reference",
            "broad_overview": "summary",
        }.get(qtype, "summary")

    try:
        confidence = float(data.get("confidence", 0.0))
        confidence = max(0.0, min(1.0, confidence))
    except Exception:
        confidence = 0.0

    result = {
        "question_type":       qtype,
        "answer_mode":         answer_mode,
        "needs_decomposition": needs_decomposition,
        "confidence":          confidence,
        "reason":              str(data.get("reason", "")).strip(),
    }
    logger.info(
        f"[CLASSIFY] type={qtype}  mode={answer_mode}"
        f"  decompose={needs_decomposition}  conf={confidence:.2f}"
        f"  reason={result['reason']!r}"
    )
    return result