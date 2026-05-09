# fetch.py  (code/retrieval/rag/fetch.py)
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_FETCH_FIELDS = [
    "doc_id", "doc_title", "section_id", "main_section_id",
    "subsection_id", "level", "title", "number", "text", "content",  # ← добавили
]


# ── ContextBlock ──────────────────────────────────────────────────────────────

@dataclass
class ContextBlock:
    text:           str
    title:          str
    number:         str
    doc_title:      str
    section_title:  str
    section_number: str
    sub_title:      str = ""
    sub_number:     str = ""
    level:          int = 1
    score:          float = 0.0
    doc_id:         str = ""

    @property
    def source(self) -> str:
        def _fmt(num: str, ttl: str) -> str:
            if not ttl:
                return f"раздел {num}" if num else ""
            if ttl.startswith(num + " ") or ttl.startswith(num + "."):
                return ttl
            return f"{num} {ttl}" if num else ttl

        sec = _fmt(self.section_number, self.section_title)
        base = f"{self.doc_title} → {sec}" if sec else self.doc_title

        if self.sub_number and self.sub_title:
            sub = _fmt(self.sub_number, self.sub_title)
            return f"{base} → {sub}"

        return base

    @property
    def formatted_text(self) -> str:
        title = self.title.strip()
        text  = self.text.strip()
        if not title and not text:
            return ""

        _num_re = re.compile(r"^\d[\d.]*\s*")

        def _strip_number(s: str) -> str:
            """Убирает ведущий числовой префикс типа '3.1 '"""
            return _num_re.sub("", s).strip()

        def _dedup_lines(s: str) -> str:
            """Убирает дублирующую вторую строку если она совпадает с первой (без учёта знаков препинания)"""
            _punct = str.maketrans("", "", ".,;:!?-–—()\"'")
            def _norm(t: str) -> str:
                return t.strip().translate(_punct).lower()
            parts = s.split("\n", 1)
            if len(parts) == 2 and _norm(parts[1]) == _norm(parts[0]):
                return parts[0].strip()
            return s

        # Чистый текст без номерного префикса
        title_clean = _strip_number(title) if title else ""
        # text может быть "3.1 Текст...\nТекст..." — убираем номер и дубль второй строки
        text_clean  = _dedup_lines(_strip_number(text)) if text else ""

        # Выбираем один содержательный body
        if text_clean and title_clean:
            if text_clean.startswith(title_clean):
                body = text_clean          # text полнее
            elif title_clean.startswith(text_clean):
                body = title_clean         # title полнее
            else:
                body = text_clean if len(text_clean) >= len(title_clean) else title_clean
        else:
            body = text_clean or title_clean

        number = self.number.strip() if self.number else ""

        # Первая строка body → заголовок ## с номером, остальное → тело
        lines = body.split("\n", 1)
        first_line = lines[0].strip()
        rest = lines[1].strip() if len(lines) > 1 else ""

        header = f"{number} {first_line}".strip() if number else first_line
        if not header:
            return ""
        return f"## {header}\n{rest}" if rest else f"## {header}"


# ── Helpers ───────────────────────────────────────────────────────────────────

def natural_sort_key(value: Any) -> List[int]:
    if value is None:
        return [999999]
    parts = re.split(r"[^\d]+", str(value))
    nums = [int(p) for p in parts if p.isdigit()]
    return nums if nums else [999999]


def _top_number(number: Any) -> str:
    if not number:
        return ""
    return str(number).split(".")[0].strip()


def _row_to_block(row: Dict, section_row: Optional[Dict] = None) -> ContextBlock:
    level     = row.get("level", 1)
    number    = str(row.get("number")    or "")
    title     = str(row.get("title")     or "")
    text = str(row.get("text") or row.get("content") or "")
    doc_title = str(row.get("doc_title") or row.get("doc_id") or "")
    doc_id    = str(row.get("doc_id")    or "")

    if level == 2 and section_row:
        return ContextBlock(
            text=text, title=title, number=number, doc_title=doc_title,
            section_title=str(section_row.get("title") or ""),
            section_number=str(section_row.get("number") or ""),
            sub_title=title, sub_number=number,
            level=level, score=row.get("score", 0.0), doc_id=doc_id,
        )
    return ContextBlock(
        text=text, title=title, number=number, doc_title=doc_title,
        section_title=title, section_number=number,
        level=level, score=row.get("score", 0.0), doc_id=doc_id,
    )


def _sort_blocks(blocks: List[ContextBlock]) -> List[ContextBlock]:
    """
    Сортирует блоки так: level=1 идёт первым, за ним его level=2 дочерние,
    отсортированные по natural_sort_key(number). Затем следующий level=1 и т.д.

    Группировка по top-номеру (первая часть до точки).
    """
    from collections import defaultdict

    # Разбиваем на группы по top-номеру раздела
    level1_order: List[str] = []
    level1_blocks: Dict[str, ContextBlock] = {}
    level2_by_top: Dict[str, List[ContextBlock]] = defaultdict(list)
    orphans: List[ContextBlock] = []  # level=2 без явного level=1

    for b in blocks:
        top = _top_number(b.number)
        if b.level == 1:
            if top not in level1_blocks:
                level1_order.append(top)
            level1_blocks[top] = b
        elif b.level == 2:
            level2_by_top[top].append(b)
        else:
            orphans.append(b)

    # Добавляем top-группы level=2, у которых нет level=1 (на случай неполных данных)
    for top in level2_by_top:
        if top not in level1_blocks and top not in level1_order:
            level1_order.append(top)

    result: List[ContextBlock] = []
    for top in level1_order:
        if top in level1_blocks:
            result.append(level1_blocks[top])
        children = sorted(level2_by_top.get(top, []), key=lambda b: natural_sort_key(b.number))
        result.extend(children)

    result.extend(orphans)
    return result


def _fetch_section_header(client, doc_id: str, main_section_id: str) -> Optional[Dict]:
    result = client.query(
        filter_expr=(
            f"doc_id == '{doc_id}' && "
            f"main_section_id == '{main_section_id}' && level == 1"
        ),
        limit=1,
        output_fields=_FETCH_FIELDS,
    )
    rows = result.get("results", [])
    return rows[0] if rows else None


def _log_blocks(blocks: List[ContextBlock], label: str = "") -> None:
    logger.info(f"[FETCH] {label}: {len(blocks)} блоков")
    for b in blocks:
        logger.info(
            f"  level={b.level}  number={b.number}"
            f"  text_len={len(b.text)}  score={b.score:.4f}"
            f"  source={b.source!r}"
        )


# ── Low-level fetch ───────────────────────────────────────────────────────────

def fetch_doc_title(client, doc_id: str) -> str:
    """Возвращает полное название документа из level=0 чанка."""
    result = client.query(
        filter_expr=f"doc_id == '{doc_id}' && level == 0",
        limit=1,
        output_fields=["doc_id", "title", "text"],
    )
    rows = result.get("results", [])
    if not rows:
        return ""
    row = rows[0]
    title = str(row.get("title") or "").strip()
    text  = str(row.get("text")  or "").strip()
    # title обычно короткий код, text — полное название
    return text or title


def fetch_doc_title_block(client, doc_id: str) -> Optional[ContextBlock]:
    """Возвращает ContextBlock level=0 с полным названием документа."""
    result = client.query(
        filter_expr=f"doc_id == '{doc_id}' && level == 0",
        limit=1,
        output_fields=_FETCH_FIELDS,
    )
    rows = result.get("results", [])
    if not rows:
        logger.info(f"[DOC TITLE] level=0 не найден для doc_id={doc_id!r}")
        return None
    row = rows[0]
    logger.info(f"[DOC TITLE] title={row.get('title','')}  text={row.get('text','')}")
    return _row_to_block(row)


def fetch_full_section(client, doc_id: str, main_section_id: str) -> List[Dict]:
    result = client.query(
        filter_expr=f"doc_id == '{doc_id}' && main_section_id == '{main_section_id}'",
        limit=200,
        output_fields=_FETCH_FIELDS,
    )
    rows = result.get("results", [])
    logger.info(f"[FETCH] raw rows count={len(rows)} numbers={[r.get('number') for r in rows]}")
    rows.sort(key=lambda x: (natural_sort_key(x.get("number")), x.get("level", 999)))
    logger.info(f"[FETCH] полный раздел main_section_id={main_section_id}: {len(rows)} строк")
    return rows


# ── Fetch для fact / definition ───────────────────────────────────────────────

def fetch_for_fact(
    client,
    hits: List[Dict],
    doc_id: str,
    important_tokens: List[str] = None,
) -> List[ContextBlock]:

    if not hits:
        return []

    doc_hits = [h for h in hits if h.get("doc_id") == doc_id]
    if not doc_hits:
        return []

    # ── группировка ──
    by_top: Dict[str, List[Dict]] = {}
    for h in doc_hits:
        top = _top_number(h.get("number"))
        by_top.setdefault(top, []).append(h)

    tops_by_score = sorted(
        by_top.keys(),
        key=lambda t: max(
            h.get("score", 0.0) for h in by_top[t] if h.get("level") == 2
        ) if any(h.get("level") == 2 for h in by_top[t]) else
        max(h.get("score", 0.0) for h in by_top[t]),
        reverse=True,
    )

    best_top = tops_by_score[0]
    best_score = max(h.get("score", 0.0) for h in by_top[best_top])

    selected_tops = [best_top]
    if len(tops_by_score) > 1:
        second_top = tops_by_score[1]
        second_score = max(h.get("score", 0.0) for h in by_top[second_top])
        if best_score - second_score < 0.05:
            selected_tops.append(second_top)

    section_hits = []
    main_sid = None
    for top in selected_tops:
        section_hits.extend(by_top[top])
        if main_sid is None:
            main_sid = by_top[top][0].get("main_section_id")

    section_hits.sort(key=lambda h: natural_sort_key(h.get("number")))

    logger.info(
        f"[FETCH] fact: разделы={selected_tops}  хитов={len(section_hits)}"
        f"  max_score={best_score:.3f}  main_sid={main_sid}"
    )

    # ── level=1 anchor → fetch полного раздела ──
    level1_hit = next((h for h in section_hits if h.get("level") == 1), None)
    if level1_hit:
        main_sid = level1_hit.get("main_section_id")
        logger.info(f"[FETCH] level=1 anchor → fetch полного раздела")

        full_rows = fetch_full_section(client, doc_id, main_sid)
        section_header = next((r for r in full_rows if r.get("level") == 1), None)

        blocks = [
            _row_to_block(
                r,
                section_row=section_header if r.get("level") == 2 else None
            )
            for r in full_rows
        ]

        blocks = _sort_blocks(blocks)
        _log_blocks(blocks, f"fact full_section (anchor)")
        return blocks

    # ── дальше без level=1 anchor ──

    if important_tokens:
        sparse_query = " ".join(important_tokens)
        for top in selected_tops:
            msid = by_top[top][0].get("main_section_id") if by_top[top] else None
            if not msid:
                continue
            sparse_hits = client.search(
                text=sparse_query,
                mode="sparse",
                level=[2],
                limit=10,
                filter_expr=f"doc_id == '{doc_id}' && main_section_id == '{msid}'",
            ).get("results", [])
            existing_numbers = {str(h.get("number") or "") for h in section_hits}
            for h in sparse_hits:
                num = str(h.get("number") or "")
                if num and num not in existing_numbers:
                    existing_numbers.add(num)
                    section_hits.append(h)

        section_hits.sort(key=lambda h: natural_sort_key(h.get("number")))

    section_headers: Dict[str, Optional[Dict]] = {}
    for top in selected_tops:
        msid = by_top[top][0].get("main_section_id") if by_top[top] else None
        if msid:
            section_headers[top] = _fetch_section_header(client, doc_id, msid)

    main_sid = by_top[selected_tops[0]][0].get("main_section_id")

    blocks: List[ContextBlock] = []
    seen_numbers: set = set()
    need_full_section = False
    need_fetch_sids: List[str] = []

    for top in selected_tops:
        sh = section_headers.get(top)
        if sh:
            num = str(sh.get("number") or "")
            if num not in seen_numbers:
                seen_numbers.add(num)
                blocks.append(_row_to_block(sh))

    def _get_header_for_hit(h: Dict) -> Optional[Dict]:
        return section_headers.get(_top_number(h.get("number")))

    for h in section_hits:
        num   = str(h.get("number") or "")
        level = h.get("level", 1)
        text  = str(h.get("text") or "").strip()
        section_header = _get_header_for_hit(h)

        if num in seen_numbers:
            continue
        seen_numbers.add(num)

        if level == 1:
            need_full_section = True
            continue
        elif level == 2 and text:
            blocks.append(_row_to_block(h, section_row=section_header))
        elif level == 2 and not text:
            sid = h.get("subsection_id") or h.get("section_id")
            if sid:
                need_fetch_sids.append(sid)

    if need_full_section and main_sid:
        full_rows = fetch_full_section(client, doc_id, main_sid)
        blocks = []
        sh = section_headers.get(selected_tops[0])
        for row in full_rows:
            blocks.append(
                _row_to_block(
                    row,
                    section_row=sh if row.get("level") == 2 else None
                )
            )
        blocks = _sort_blocks(blocks)
        return blocks

    if need_fetch_sids:
        sids_expr = ", ".join(f"'{s}'" for s in need_fetch_sids)
        result = client.query(
            filter_expr=f"doc_id == '{doc_id}' && subsection_id in [{sids_expr}]",
            limit=50,
            output_fields=_FETCH_FIELDS,
        )
        for row in result.get("results", []):
            rnum = str(row.get("number") or "")
            if rnum not in seen_numbers:
                seen_numbers.add(rnum)
                sh = section_headers.get(_top_number(row.get("number")))
                blocks.append(_row_to_block(row, section_row=sh))

    blocks = _sort_blocks(blocks)
    _log_blocks(blocks, f"fact doc={doc_id}")
    return blocks


# ── Fetch для section ─────────────────────────────────────────────────────────

def fetch_for_section(client, doc_id: str, main_section_id: str) -> List[ContextBlock]:
    rows = fetch_full_section(client, doc_id, main_section_id)
    if not rows:
        return []

    section_header = next((r for r in rows if r.get("level") == 1), None)

    blocks = [
        _row_to_block(r, section_row=section_header if r.get("level") == 2 else None)
        for r in rows
    ]

    blocks = _sort_blocks(blocks)
    _log_blocks(blocks, f"section main_sid={main_section_id}")
    return blocks