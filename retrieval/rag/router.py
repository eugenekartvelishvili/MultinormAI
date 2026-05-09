# router.py  (code/retrieval/rag/router.py)
from __future__ import annotations

import argparse
import json
import logging
import re
from typing import Any, Dict, List

from code.retrieval.rag.answer_generator import generate_answer
from code.retrieval.rag.classifier import classify_question
from code.retrieval.rag.context import build_context, build_sources, format_sources_bbcode
from code.retrieval.rag.fetch import ContextBlock
from code.retrieval.rag.milvus_client import MilvusApiClient
from code.retrieval.rag.rewriter import rewrite_query
from code.retrieval.rag.strategies import strategy_a, strategy_b, strategy_c

logger = logging.getLogger(__name__)


# ── BBCode форматирование ─────────────────────────────────────────────────────

def _convert_md_table(match: re.Match) -> str:
    """Конвертирует markdown таблицу в BBCode [TABLE]."""
    block = match.group(0).strip()
    lines = [l.strip() for l in block.splitlines() if l.strip()]

    result = ["[TABLE]"]
    is_header = True

    for line in lines:
        # Пропускаем разделительную строку (|---|---|)
        if re.match(r"^\|[\s\-\|:]+\|$", line):
            is_header = False
            continue

        # Разбиваем строку на ячейки
        cells = [c.strip() for c in line.strip("|").split("|")]

        if is_header:
            row = "[TR]" + "".join(f"[TH]{c}[/TH]" for c in cells) + "[/TR]"
            is_header = False
        else:
            row = "[TR]" + "".join(f"[TD]{c}[/TD]" for c in cells) + "[/TR]"

        result.append(row)

    result.append("[/TABLE]")
    return "\n".join(result)


def markdown_to_bbcode(text: str) -> str:
    # Таблицы — конвертируем до остальных правил
    text = re.sub(
        r"(\|.+\|\n)([\|\s\-\:]+\|\n)(\|.+\|\n?)+",
        _convert_md_table,
        text,
    )

    text = re.sub(r"^#{1,6}\s*(.+)$", r"[B]\1[/B]", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*(.+?)\*\*", r"[B]\1[/B]", text)
    text = re.sub(r"__(.+?)__", r"[B]\1[/B]", text)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"[I]\1[/I]", text)
    text = re.sub(r"(?<!_)_(?!_)(.+?)(?<!_)_(?!_)", r"[I]\1[/I]", text)
    text = re.sub(r"~~(.+?)~~", r"[S]\1[/S]", text)
    text = re.sub(r"```[\w]*\n?", "", text)
    text = re.sub(r"`(.+?)`", r"\1", text)
    text = re.sub(r"\[(.+?)\]\((.+?)\)", r"[URL=\2]\1[/URL]", text)
    text = re.sub(r"^[\*\-]\s+", "• ", text, flags=re.MULTILINE)
    text = re.sub(r"^>\s?(.+)$", r">>\1", text, flags=re.MULTILINE)
    text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── Получение полного названия документа ─────────────────────────────────────

def _fetch_full_doc_title(client, doc_id: str) -> str:
    """Запрашивает полное название документа из level=0 чанка по doc_id."""
    try:
        rows = client.query(
            filter_expr=f"doc_id == '{doc_id}' && level == 0",
            limit=1,
            output_fields=["doc_id", "title", "text"],
        ).get("results", [])
        logger.info(f"[DOC TITLE] запрос по doc_id={doc_id!r}  найдено={len(rows)} строк")
        if rows:
            logger.info(f"[DOC TITLE] raw title={rows[0].get('title','')[:200]!r}  raw text={rows[0].get('text','')[:200]!r}")
            full = str(rows[0].get("text") or rows[0].get("title") or "").strip()
            if full:
                return full
    except Exception as e:
        logger.warning(f"[DOC TITLE] ошибка запроса: {e}")
    return ""


# ── Главный flow ──────────────────────────────────────────────────────────────

def adaptive_search(question: str) -> Dict[str, Any]:
    import time
    t0 = time.time()

    logger.info(f"[RAG] ── Новый запрос ──────────────────────────────────")
    logger.info(f"[RAG] Вопрос: {question!r}")

    # 1. Классификация
    t1 = time.time()
    classification = classify_question(question)
    logger.info(f"[TIME] classify={time.time()-t1:.2f}s")

    question_type = classification["question_type"]
    answer_mode = classification["answer_mode"]
    needs_decomposition = classification["needs_decomposition"]

    # 2. Переформулировка + декомпозиция + токены
    t2 = time.time()
    rewrite = rewrite_query(question, needs_decomposition)
    logger.info(f"[TIME] rewrite={time.time()-t2:.2f}s")

    rewritten = rewrite["rewritten"]
    important_tokens = rewrite["important_tokens"]
    entity = rewrite.get("entity", rewritten)
    attribute = rewrite.get("attribute", rewritten)

    if important_tokens:
        logger.info(f"[TOKENS] {important_tokens}")

    # 3. Выбор стратегии и поиск
    client = MilvusApiClient()

    t3 = time.time()

    if question_type == "norm_reference":
        logger.info(f"[SEARCH] стратегия C (norm_reference)")
        blocks: List[ContextBlock] = strategy_c(client, rewritten, important_tokens)

    elif question_type in ("procedure", "broad_overview"):
        logger.info(f"[SEARCH] стратегия B ({question_type})")
        blocks = strategy_b(client, entity, attribute, important_tokens)

    else:
        logger.info(f"[SEARCH] стратегия A ({question_type})")
        blocks = strategy_a(client, rewritten, important_tokens)

    logger.info(f"[TIME] search+fetch={time.time()-t3:.2f}s")

    # 4. Сборка контекста и источников из единого списка блоков
    context = build_context(blocks)
    sources = build_sources(blocks, max_sources=3)

    logger.info(f"[CONTEXT] итоговый контекст: {len(context)} символов")
    logger.info(f"[SOURCES] финальные источники: {sources}")

    # 5. Генерация ответа
    t4 = time.time()
    if context.strip():
        logger.info(f"[ANSWER] генерируем ответ, контекст {len(context)} символов")
        logger.info(f"[ANSWER] контекст ({len(context)} символов):\n{context}")
        answer = generate_answer(
            question=question,
            context=context,
            question_type=question_type,
            answer_mode=answer_mode,
        )
    else:
        logger.info("[ANSWER] контекст пуст")
        answer = "В документах нет информации по этому вопросу."

    logger.info(f"[TIME] answer={time.time()-t4:.2f}s")

    answer_bbcode = markdown_to_bbcode(answer.strip())
    answer_bbcode += "\n\n" + format_sources_bbcode(sources)

    logger.info(f"[TIME] total={time.time()-t0:.2f}s")
    logger.info(f"[RAG] ── Готово ──────────────────────────────────────")

    return {
        "question":       question,
        "search_query":   rewritten,
        "question_type":  question_type,
        "answer_mode":    answer_mode,
        "classification": classification,
        "results":        [{"source": b.source, "score": b.score, "number": b.number} for b in blocks],
        "context":        context,
        "answer":         answer_bbcode,
        "sources":        sources,
    }