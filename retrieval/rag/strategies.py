# strategies.py  (code/retrieval/rag/strategies.py)
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

from code.retrieval.rag.config import (
    RAG_TOP_SECTIONS,
    SCORE_DOC_GAP_THRESHOLD,
    SCORE_SECTION_GAP_THRESHOLD,
    SCORE_SECTION_GAP_THRESHOLD_B,
    SCORE_SUMMARY_CONFIDENT,
)
from code.retrieval.rag.fetch import (
    ContextBlock,
    fetch_doc_title_block,
    fetch_for_fact,
    fetch_for_section,
    fetch_full_section,
    _row_to_block,
    natural_sort_key,
    _top_number,
)

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _log_hits(hits: List[Dict], label: str) -> None:
    logger.info(f"[SEARCH] {label}: {len(hits)} хитов")
    for i, h in enumerate(hits):
        logger.info(
            f"  [{i+1}] score={h.get('score', 0):.4f}"
            f"  level={h.get('level')}  number={h.get('number')}"
            f"  doc_id={h.get('doc_id')}  title={h.get('title', '')!r}"
        )


def _pick_doc_id(hits: List[Dict]) -> tuple[str | None, Dict | None]:
    """
    Выбор документа: агрегируем все хиты по doc_id,
    score документа = max(dense+summary) среди его хитов.
    Бонус +0.1 если есть level=0 хит.

    Якорный раздел = лучший level=2 хит документа (конкретный подпункт).
    Если level=2 нет — лучший level=1 хит.
    """
    best_by_doc: Dict[str, Dict] = {}
    anchor_by_doc: Dict[str, Dict] = {}  # лучший level=2 хит для каждого документа

    for h in hits:
        doc_id = h.get("doc_id")
        if not doc_id:
            continue
        eff = (h.get("_dense", 0.0) or 0.0) + (h.get("_summary", 0.0) or 0.0)
        level = h.get("level", 1)

        # Обновляем лучший хит документа (с бонусом за level=0)
        eff_with_bonus = eff + (0.1 if level == 0 else 0.0)
        if doc_id not in best_by_doc or eff_with_bonus > best_by_doc[doc_id]["_eff"]:
            best_by_doc[doc_id] = {**h, "_eff": eff_with_bonus}

        # Обновляем якорный level=2 хит документа
        if level == 2:
            if doc_id not in anchor_by_doc or eff > anchor_by_doc[doc_id].get("_eff2", 0.0):
                anchor_by_doc[doc_id] = {**h, "_eff2": eff}

    if not best_by_doc:
        return None, None

    best = max(best_by_doc.values(), key=lambda x: x["_eff"])
    doc_id = best.get("doc_id")

    # Якорь — лучший level=2 хит документа
    anchor = anchor_by_doc.get(doc_id)

    # Если level=2 нет — берём лучший level=1 хит (но не level=0, у него нет main_section_id)
    if anchor is None:
        level1_candidates = [
            h for h in hits
            if h.get("doc_id") == doc_id and h.get("level") == 1 and h.get("main_section_id")
        ]
        if level1_candidates:
            anchor = max(
                level1_candidates,
                key=lambda h: (h.get("_dense", 0.0) or 0.0) + (h.get("_summary", 0.0) or 0.0)
            )

    logger.info(
        f"[DOC] выбран doc_id={doc_id}"
        f"  eff={best['_eff']:.4f}  level={best.get('level')}"
        f"  (d={best.get('_dense',0):.2f} sum={best.get('_summary',0):.2f})"
        f"  anchor={anchor.get('number') if anchor else None}"
        f"  anchor_title={anchor.get('title', '')[:50] if anchor else 'None'}"
    )
    return doc_id, anchor


def _pick_top_section(hits: List[Dict]) -> str | None:
    """Возвращает main_section_id раздела с наибольшим суммарным score хитов."""
    by_top: Dict[str, List[Dict]] = {}
    for h in hits:
        top = _top_number(h.get("number"))
        by_top.setdefault(top, []).append(h)

    if not by_top:
        return None

    best_top = max(by_top, key=lambda t: sum(h.get("score", 0) for h in by_top[t]))
    best_hits = by_top[best_top]
    main_sid = best_hits[0].get("main_section_id") if best_hits else None
    logger.info(f"[SECTION] топ раздел={best_top}  main_sid={main_sid}")
    return main_sid


# ── Стратегия A: fact / definition ───────────────────────────────────────────

def strategy_a(
    client,
    query: str,
    important_tokens: List[str],
) -> List[ContextBlock]:
    """
    Pass 1: глобальный max_score поиск → выбираем документ.
    Pass 2: max_score по level=1 внутри документа → находим правильный раздел.
    Pass 3: fetch раздела + sparse по important_tokens для уточнения.
    """
    logger.info(f"[STRATEGY A] query={query!r}  tokens={important_tokens}")

    # Pass 1 — находим документ
    hits = client.search(
        text=query,
        mode="max_score",
        level=[0, 1, 2],
        limit=15,
    ).get("results", [])
    _log_hits(hits, "pass1 global")

    if not hits:
        return []

    # Выбираем документ по лучшему хиту (dense+summary, без sparse)
    doc_id, anchor_hit = _pick_doc_id(hits)
    if not doc_id:
        return []

    # Получаем полное название документа
    title_block = fetch_doc_title_block(client, doc_id)

    anchor_main_sid = anchor_hit.get("main_section_id") if anchor_hit else None
    logger.info(f"[STRATEGY A] anchor: number={anchor_hit.get('number') if anchor_hit else None}  main_sid={anchor_main_sid}")

    # Pass 2 — ищем раздел по level=1 внутри документа
    section_hits = client.search(
        text=query,
        mode="max_score",
        level=[1],
        limit=5,
        filter_expr=f"doc_id == '{doc_id}'",
    ).get("results", [])
    _log_hits(section_hits, f"pass2 sections doc={doc_id}")

    if not section_hits:
        logger.info("[STRATEGY A] pass2 пустой, fallback на глобальные хиты")
        doc_hits = [h for h in hits if h.get("doc_id") == doc_id]
        blocks = fetch_for_fact(client, doc_hits, doc_id, important_tokens=important_tokens)
        return ([title_block] + blocks) if title_block else blocks

    # Берём топ-1 раздел (и топ-2 если gap < порога)
    best_section = section_hits[0]
    best_main_sid = best_section.get("main_section_id")
    best_score = best_section.get("score", 0.0)

    selected_main_sids = [best_main_sid] if best_main_sid else []

    if len(section_hits) > 1:
        second = section_hits[1]
        second_main_sid = second.get("main_section_id")
        second_score = second.get("score", 0.0)
        if second_main_sid and second_main_sid not in selected_main_sids and best_score - second_score < SCORE_SECTION_GAP_THRESHOLD:
            selected_main_sids.append(second_main_sid)
            logger.info(
                f"[STRATEGY A] два раздела:"
                f"  топ-1={best_section.get('number')}({best_score:.3f})"
                f"  топ-2={second.get('number')}({second_score:.3f})"
                f"  gap={best_score-second_score:.3f}"
            )

    # Гарантируем что раздел anchor хита попадёт
    if anchor_main_sid and anchor_main_sid not in selected_main_sids:
        selected_main_sids.append(anchor_main_sid)
        logger.info(f"[STRATEGY A] добавлен anchor раздел main_sid={anchor_main_sid}")

    logger.info(f"[STRATEGY A] выбраны разделы main_sids={selected_main_sids}")

    # Собираем level=2 хиты из выбранных разделов
    doc_hits = [h for h in hits if h.get("doc_id") == doc_id and
                h.get("main_section_id") in selected_main_sids]

    # Если level=2 хитов нет — добавляем level=1 хит чтобы fetch взял раздел целиком
    if not any(h.get("level") == 2 for h in doc_hits):
        doc_hits.extend([h for h in section_hits if h.get("main_section_id") in selected_main_sids])

    blocks = fetch_for_fact(client, doc_hits, doc_id, important_tokens=important_tokens)
    return ([title_block] + blocks) if title_block else blocks


# ── Стратегия B: procedure / broad_overview ───────────────────────────────────

def strategy_b(
    client,
    entity_query: str,
    attribute_query: str,
    important_tokens: List[str],
) -> List[ContextBlock]:
    """
    Pass 1: ищем документ по summary (level=0+1).
    Pass 2: ищем разделы внутри документа по level=1.
    Fetch: берём топ разделы целиком, отсекая по SCORE_SECTION_GAP_THRESHOLD_B.
    """
    logger.info(f"[STRATEGY B] entity={entity_query!r}  attribute={attribute_query!r}")

    # Pass 1 — документ по summary
    pass1_hits = client.search(
        text=entity_query,
        mode="hybrid",
        level=[0, 1],
        limit=8,
        use_summary=True,
        dense_weight=0.3,
        summary_weight=0.65,
        sparse_weight=0.05,
    ).get("results", [])
    _log_hits(pass1_hits, "pass1 summary")

    top_score = pass1_hits[0].get("score", 0.0) if pass1_hits else 0.0
    if top_score < SCORE_SUMMARY_CONFIDENT:
        logger.info(f"[STRATEGY B] summary слабый ({top_score:.4f}), fallback по тексту")
        pass1_hits = client.search(
            text=entity_query,
            mode="max_score",
            level=[0, 1],
            limit=8,
        ).get("results", [])
        _log_hits(pass1_hits, "pass1 fallback")

    doc_id, anchor_hit = _pick_doc_id(pass1_hits)
    if not doc_id:
        logger.info("[STRATEGY B] документ не найден, fallback → A")
        return strategy_a(client, attribute_query, important_tokens)

    # Получаем полное название документа
    title_block = fetch_doc_title_block(client, doc_id)

    anchor_main_sid = anchor_hit.get("main_section_id") if anchor_hit else None
    anchor_score = pass1_hits[0].get("score", 0.0) if pass1_hits else 0.0
    second_score = pass1_hits[1].get("score", 0.0) if len(pass1_hits) > 1 else 0.0
    anchor_gap = anchor_score - second_score

    logger.info(f"[STRATEGY B] anchor: number={anchor_hit.get('number') if anchor_hit else None}  main_sid={anchor_main_sid}  gap={anchor_gap:.3f}")

    # Если anchor уверенный — берём только его раздел, pass2 не нужен
    if anchor_main_sid and anchor_gap > SCORE_SECTION_GAP_THRESHOLD_B:
        logger.info(f"[STRATEGY B] anchor уверенный (gap={anchor_gap:.3f}), берём только раздел pass1")
        return fetch_for_section(client, doc_id, anchor_main_sid)

    # Pass 2 — разделы внутри документа, только level=1
    pass2_hits = client.search(
        text=attribute_query,
        mode="max_score",
        level=[1],
        limit=10,
        filter_expr=f"doc_id == '{doc_id}'",
    ).get("results", [])
    _log_hits(pass2_hits, f"pass2 sections doc={doc_id}")

    if not pass2_hits:
        logger.info("[STRATEGY B] pass2 пустой, fallback → A")
        return strategy_a(client, attribute_query, important_tokens)

    # Группируем по top-номеру, сортируем по max score
    by_top: Dict[str, List[Dict]] = {}
    for h in pass2_hits:
        top = _top_number(h.get("number"))
        by_top.setdefault(top, []).append(h)

    sorted_tops = sorted(
        by_top.keys(),
        key=lambda t: max(h.get("score", 0) for h in by_top[t]),
        reverse=True,
    )

    best_score = max(h.get("score", 0) for h in by_top[sorted_tops[0]])
    logger.info(f"[STRATEGY B] топ раздел={sorted_tops[0]}  score={best_score:.4f}")

    # Собираем разделы с gap-отсечением по pass2
    blocks: List[ContextBlock] = []
    seen_sections: set = set()

    for top in sorted_tops[:RAG_TOP_SECTIONS]:
        top_score = max(h.get("score", 0) for h in by_top[top])
        gap = best_score - top_score

        if gap > SCORE_SECTION_GAP_THRESHOLD_B:
            logger.info(
                f"[STRATEGY B] раздел {top} отсечён по gap"
                f"  ({best_score:.3f} - {top_score:.3f} = {gap:.3f} > {SCORE_SECTION_GAP_THRESHOLD_B})"
            )
            break

        main_sid = by_top[top][0].get("main_section_id")
        if not main_sid or main_sid in seen_sections:
            continue
        seen_sections.add(main_sid)

        logger.info(f"[STRATEGY B] берём раздел {top} из pass2  score={top_score:.4f}  main_sid={main_sid}")
        section_blocks = fetch_for_section(client, doc_id, main_sid)
        blocks.extend(section_blocks)

    # Anchor из pass1 — добавляем первым если pass2 его не взял
    if anchor_main_sid and anchor_main_sid not in seen_sections:
        logger.info(f"[STRATEGY B] добавляем anchor раздел первым main_sid={anchor_main_sid}")
        anchor_blocks = fetch_for_section(client, doc_id, anchor_main_sid)
        blocks = anchor_blocks + blocks
    elif not blocks and anchor_main_sid:
        logger.info(f"[STRATEGY B] pass2 пустой, fallback на anchor main_sid={anchor_main_sid}")
        blocks = fetch_for_section(client, doc_id, anchor_main_sid)

    return ([title_block] + blocks) if title_block else blocks


# ── Стратегия C: norm_reference ───────────────────────────────────────────────

def strategy_c(
    client,
    query: str,
    important_tokens: List[str],
) -> List[ContextBlock]:
    """
    Pass 1: документ по summary.
    Pass 2: раздел внутри документа.
    Возвращаем раздел целиком.
    """
    logger.info(f"[STRATEGY C] query={query!r}")

    # Pass 1 — документ по summary
    pass1_hits = client.search(
        text=query,
        mode="hybrid",
        level=[0, 1],
        limit=8,
        use_summary=True,
        dense_weight=0.3,
        summary_weight=0.65,
        sparse_weight=0.05,
    ).get("results", [])
    _log_hits(pass1_hits, "pass1 doc summary")

    if not pass1_hits:
        logger.info("[STRATEGY C] pass1 пустой, fallback → A")
        return strategy_a(client, query, important_tokens)

    top_score = pass1_hits[0].get("score", 0.0)
    if top_score < SCORE_SUMMARY_CONFIDENT:
        logger.info(f"[STRATEGY C] summary слабый ({top_score:.4f}), fallback → A")
        return strategy_a(client, query, important_tokens)

    doc_id, _ = _pick_doc_id(pass1_hits)
    if not doc_id:
        logger.info("[STRATEGY C] документ не найден, fallback → A")
        return strategy_a(client, query, important_tokens)

    logger.info(f"[STRATEGY C] doc_id={doc_id}  score={top_score:.4f}")

    # Получаем полное название документа
    title_block = fetch_doc_title_block(client, doc_id)

    def _with_title(blocks):
        return ([title_block] + blocks) if title_block else blocks

    # Pass 2 — раздел внутри документа
    pass2_hits = client.search(
        text=query,
        mode="max_score",
        level=[1],
        limit=5,
        filter_expr=f"doc_id == '{doc_id}'",
    ).get("results", [])
    _log_hits(pass2_hits, f"pass2 section doc={doc_id}")

    if not pass2_hits:
        logger.info("[STRATEGY C] pass2 пустой, fallback → A")
        return strategy_a(client, query, important_tokens)

    # Берём топ-1 раздел (и топ-2 если gap маленький) — только заголовок, без подпунктов
    best_score = pass2_hits[0].get("score", 0.0)
    seen_sections: set = set()
    blocks: List[ContextBlock] = []

    for h in pass2_hits[:RAG_TOP_SECTIONS]:
        h_score = h.get("score", 0.0)
        if best_score - h_score > SCORE_SECTION_GAP_THRESHOLD_B:
            logger.info(f"[STRATEGY C] раздел {h.get('number')} отсечён по gap ({best_score:.3f} - {h_score:.3f})")
            break
        msid = h.get("main_section_id")
        if not msid or msid in seen_sections:
            continue
        seen_sections.add(msid)
        logger.info(f"[STRATEGY C] берём раздел {h.get('number')}  score={h_score:.4f}  main_sid={msid}")
        # Для norm_reference достаточно только заголовка раздела level=1
        blocks.append(_row_to_block(h))

    return _with_title(blocks)