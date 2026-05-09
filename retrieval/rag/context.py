# context.py  (code/retrieval/rag/context.py)
from __future__ import annotations

import logging
from typing import List

from code.retrieval.rag.fetch import ContextBlock

logger = logging.getLogger(__name__)


def build_context(blocks: List[ContextBlock]) -> str:
    """
    Собирает текст контекста из списка ContextBlock.
    Если есть блок level=0 — берёт из него полное название документа и ставит в начало.
    """
    # Полное название документа из level=0 блока
    l0 = next((b for b in blocks if b.level == 0), None)
    if l0:
        doc_title = (l0.text or l0.title or l0.doc_title or "").strip()
    else:
        doc_title = blocks[0].doc_title.strip() if blocks else ""

    parts = []
    for b in blocks:
        if b.level == 0:
            continue  # не выводим как блок — используем только для заголовка
        text = b.formatted_text
        if text:
            parts.append(text)

    context = "\n\n".join(parts)

    if doc_title:
        context = f"Документ: {doc_title}\n\n{context}"

    logger.info(f"[CONTEXT] {len(context)} символов, {len(parts)} блоков")
    logger.info(f"[CONTEXT FULL]\n{context}")
    return context


def build_sources(blocks: List[ContextBlock], max_sources: int = 3) -> List[str]:
    """
    Строит список уникальных источников из блоков контекста.
    Порядок: как в контексте (не по score).
    Дедупликация до уровня раздела — если уже есть раздел, подпункт не добавляем.
    """
    seen: set = set()
    sources: List[str] = []

    for b in blocks:
        if b.level == 0:
            continue
        src = b.source
        if not src:
            continue

        if b.level == 1:
            section_src = f"{b.doc_title} → {b.section_title}" if b.section_title else b.doc_title
            if section_src not in seen:
                seen.add(section_src)
                seen.add(src)
                sources.append(src)
        else:
            if src not in seen:
                seen.add(src)
                sources.append(src)

        if len(sources) >= max_sources:
            break

    logger.info(f"[SOURCES] {sources}")
    return sources


def format_sources_bbcode(sources: List[str]) -> str:
    if not sources:
        return "[B]Источники:[/B] не найдены"
    lines = ["[B]Источники:[/B]"] + [f"• {s}" for s in sources]
    return "\n".join(lines)