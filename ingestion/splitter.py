from bs4 import BeautifulSoup, Tag
from typing import List
from code.ingestion.models import HtmlSection
import uuid

def split_html_by_h2(*, doc_id: str, html: str, base_metadata: dict, max_chunk_size: int = 3000) -> List[HtmlSection]:
    """
    Разбивает HTML по <h2>. Если секция слишком большая, делим на части по <h3>.
    Строгое правило: минимум 2 H3 на чанк.
    """
    soup = BeautifulSoup(html, "lxml")
    result: List[HtmlSection] = []

    h2_list = soup.find_all("h2")

    for idx, h2 in enumerate(h2_list):
        h2_title = h2.get_text(strip=True)
        main_section_id: str | None = None
        chunk_index = 0

        # --- собираем всё до следующего h2 ---
        nodes = []
        for sib in h2.next_siblings:
            if isinstance(sib, Tag) and sib.name == "h2":
                break
            if isinstance(sib, str) and not sib.strip():
                continue
            nodes.append(sib)

        # print("\n" + "=" * 100)
        # print(f"H2[{idx}] {h2_title}")
        # print(f"Top-level nodes: {[n.name if isinstance(n, Tag) else 'TEXT' for n in nodes]}")

        # --- разворачиваем div.h-block, если есть ---
        if len(nodes) == 1 and isinstance(nodes[0], Tag) and nodes[0].name == "div":
            # print(">> unwrap div.h-block")
            nodes = [n for n in nodes[0].children if not (isinstance(n, str) and not n.strip())]

        if not nodes:
            # print(">> EMPTY SECTION")
            section_id = str(uuid.uuid4())
            main_section_id = section_id
            result.append(HtmlSection(
                doc_id=doc_id,
                section_id=section_id,
                main_section_id=main_section_id,
                title=h2_title,
                html="",
                metadata={**base_metadata, "section_index": idx, "title": h2_title},
            ))
            continue

        full_html = "".join(str(n) for n in nodes)

        # --- маленькая секция целиком ---
        if len(full_html) <= max_chunk_size:
            # print(f">> SINGLE CHUNK ({len(full_html)} chars)")
            section_id = str(uuid.uuid4())
            main_section_id = section_id
            result.append(HtmlSection(
                doc_id=doc_id,
                section_id=section_id,
                main_section_id=main_section_id,
                title=h2_title,
                html=full_html,
                metadata={**base_metadata, "section_index": idx, "chunk_index": 0, "title": h2_title},
            ))
            continue

        # --- большая секция → разбиваем на H3-группы ---
        # print(">> SPLIT BY H3")
        h3_groups: List[List[Tag]] = []
        current_group: List[Tag] = []

        for n in nodes:
            if isinstance(n, Tag) and n.name == "h3":
                if current_group:
                    h3_groups.append(current_group)
                current_group = [n]
            else:
                current_group.append(n)

        if current_group:
            h3_groups.append(current_group)

        # print(f">> Found {len(h3_groups)} H3 groups")

        # --- теперь формируем чанки с минимум 2 H3 ---
        chunk: List[Tag] = []
        chunk_size = 0
        chunk_index = 0
        i = 0
        while i < len(h3_groups):
            chunk.append(h3_groups[i])
            chunk_size += len("".join(str(n) for n in h3_groups[i]))
            h3_count_in_chunk = 1

            # добавляем второй H3, если есть, или предыдущий, если последний
            if i + 1 < len(h3_groups):
                chunk.append(h3_groups[i + 1])
                chunk_size += len("".join(str(n) for n in h3_groups[i + 1]))
                h3_count_in_chunk += 1
                i += 1
            elif h3_count_in_chunk == 1 and chunk_index > 0:
                # последний H3, присоединяем к предыдущему чанку
                result[-1].html += "".join(str(n) for n in chunk)
                i += 1
                continue

            # добавляем следующие H3, пока не превысим размер
            j = i + 1
            while j < len(h3_groups) and chunk_size + len("".join(str(n) for n in h3_groups[j])) <= max_chunk_size:
                chunk.append(h3_groups[j])
                chunk_size += len("".join(str(n) for n in h3_groups[j]))
                h3_count_in_chunk += 1
                j += 1

            chunk_html = "".join(str(n) for n in chunk)
            section_id = str(uuid.uuid4())
            if main_section_id is None:
                main_section_id = section_id

            # print(f"\n--- OUTPUT CHUNK ({chunk_size} chars, {h3_count_in_chunk} H3) ---")
            result.append(HtmlSection(
                doc_id=doc_id,
                section_id=section_id,
                main_section_id=main_section_id,
                title=h2_title,
                html=chunk_html,
                metadata={**base_metadata, "section_index": idx, "chunk_index": chunk_index, "title": h2_title},
            ))

            chunk_index += 1
            chunk = []
            chunk_size = 0
            i = j

    return result
