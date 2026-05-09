# code/retrieval/rag/rewriter.py
from __future__ import annotations

import json
import logging
import re
from typing import Dict, Optional

from langchain_ollama import ChatOllama
from code.retrieval.rag.config import LOCAL_LLM_MODEL, LLM_BASE_URL

logger = logging.getLogger(__name__)


def _get_llm(llm: Optional[ChatOllama] = None) -> ChatOllama:
    if llm is not None:
        return llm
    return ChatOllama(model=LOCAL_LLM_MODEL, temperature=0.0, base_url=LLM_BASE_URL)


_PROMPT = """Ты помогаешь подготовить поисковые запросы по корпоративным нормативным документам.

Вопрос: "{question}"

Задача 1 — переформулируй вопрос:
- Исправь опечатки
- Замени разговорные слова официальной терминологией
  Примеры: "уволиться" → "расторжение трудового договора", "заявка на тендер" → "тендерное предложение"
- НЕ добавляй ГОСТы и стандарты
- НЕ меняй аббревиатуры (ПТП, ГООС, ИСМ и т.д.)
- НЕ расшифровывай однобуквенные обозначения (диск P, диск F — оставляй как есть)
- Результат не длиннее оригинала в 2 раза

Задача 2 — выдели важные токены для точного поиска:
- Специфичные термины которые должны встретиться дословно в тексте
- Аббревиатуры, коды документов (ИН-01-ВП, ПР-6.1.2), классы (IV класс)
- Ключевые понятия вопроса (финансовые риски, переоценка поставщиков)
- Максимум 3-5 токенов, только самые важные

Задача 3 — декомпозиция:
- entity: конкретная сущность о которой вопрос (отдел, объект, документ, процесс)
  ВАЖНО: entity должен быть полной значимой фразой, не одним словом
- attribute: что именно про неё ищем (задачи, структура, порядок, требования)
Примеры:
- "Какие задачи у отдела геологии" → entity="отдел геологии", attribute="задачи"
- "Как транспортировать отходы IV класса" → entity="отходы IV класса", attribute="транспортирование требования"
- "Какая структура у ГООС" → entity="ГООС", attribute="структура состав"
- "Какие у нас стадии выполнения работ" → entity="стадии выполнения работ", attribute="перечень этапы"
- "Как проходит процедура закупки оборудования" → entity="закупка оборудования", attribute="процедура"

Верни ТОЛЬКО JSON:
{{
  "rewritten": "переформулированный вопрос",
  "important_tokens": ["токен1", "токен2"],
  "entity": "сущность",
  "attribute": "атрибут"
}}"""


def _extract_json(text: str) -> dict:
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
    return {}


def rewrite_query(
    question: str,
    needs_decomposition: bool,
    llm: Optional[ChatOllama] = None,
) -> Dict:

    # Если декомпозиция не нужна — не вызываем LLM, возвращаем оригинал
    if not needs_decomposition:
        logger.info(f"[REWRITE] no decomposition → original={question!r}  tokens=[]")
        return {
            "rewritten":        question,
            "important_tokens": [],
        }

    _llm = _get_llm(llm)
    prompt = _PROMPT.format(question=question)

    try:
        response = _llm.invoke([{"role": "user", "content": prompt}])
        data = _extract_json(response.content)
    except Exception as e:
        logger.warning(f"[REWRITE] ошибка LLM: {e}, используем оригинал")
        data = {}

    rewritten = str(data.get("rewritten") or question).strip()

    if len(rewritten) > len(question) * 2.5:
        rewritten = question
    forbidden = ["гост", "iso", "стандарт"]
    if any(f in rewritten.lower() for f in forbidden):
        rewritten = question

    tokens = data.get("important_tokens") or []
    if isinstance(tokens, list):
        tokens = [str(t).strip() for t in tokens if t][:5]
    else:
        tokens = []

    entity    = str(data.get("entity")    or rewritten).strip()
    attribute = str(data.get("attribute") or rewritten).strip()

    logger.info(
        f"[REWRITE] rewritten={rewritten!r}"
        f"  entity={entity!r}  attribute={attribute!r}"
        f"  tokens={tokens}"
    )

    return {
        "rewritten":        rewritten,
        "important_tokens": tokens,
        "entity":           entity,
        "attribute":        attribute,
    }