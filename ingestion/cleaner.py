import re
from collections import Counter
from bs4 import BeautifulSoup, Tag
from typing import List, Tuple
from code.ingestion.models import HtmlSection
from pathlib import Path
# -------------------------
def remove_headers_footers_early(html_text: str) -> str:
    soup = BeautifulSoup(html_text, "html.parser")

    # --- Шаг -1. Удаляем недекодированные формулы ---
    for el in soup.find_all("div", class_="formula-not-decoded"):
        el.decompose()

    path_pattern = re.compile(r'(F:\\[^\s\n\r]+)', re.I)

    for h2 in soup.find_all("h2"):
        text = h2.get_text()
        
        # если h2 содержит путь (т.е. ложный заголовок), превращаем его в <p>
        if path_pattern.search(text):
            new_html = path_pattern.sub(
                lambda m: f'<a href="{m.group(1)}">{m.group(1)}</a>',
                text
            )
            # создаём <p> и вставляем туда содержимое
            new_tag = soup.new_tag("p")
            fragment = BeautifulSoup(new_html, "lxml")
            for child in fragment.body.contents if fragment.body else fragment.contents:
                new_tag.append(child)
            # заменяем h2 на новый тег
            h2.replace_with(new_tag)



    # --- Шаг 0. Приложения делаем h2 ---
    for el in soup.find_all(["p", "div", "span", "li"]):
        text = el.get_text(strip=True)
        if re.match(r'^Приложение\s+\S+', text, re.I):
            h_tag = soup.new_tag("h2")
            h_tag.string = text
            el.replace_with(h_tag)

    # --- Шаг 1. Собираем все текстовые блоки ---
    blocks = [el.get_text(strip=True) for el in soup.find_all(["p", "div", "span", "li"])]
    counter = Counter(blocks)

#    repetitive = {text for text, cnt in counter.items() if cnt > 5 and len(text) < 120}

#    stop_words = [
#        "редакция", "всего листов", "номер документа",
#        "лист", "страница", "page", "стр. ", "из ",
#        "подпись", "дата", "код", "Рег. №"
#    ]
#    stop_pattern = re.compile("|".join(stop_words), re.I)

#    for el in soup.find_all("p"):
#        text = el.get_text(strip=True)
#        if (text in repetitive or stop_pattern.search(text)) and not re.search(r"утверждаю|соглас", text, re.I):
#            el.decompose()

    return str(soup)

def is_real_heading(text: str) -> bool:
    # text = text.strip()

    # if len(text) > 100:
    #     return False

    # # if text.endswith(":"):
    # #     return False

    # verb_pattern = re.compile(
    #     r"\b(направля|осуществля|проводит|производит|являет|предусматрива|определя)\w*",
    #     re.I
    # )
    # if verb_pattern.search(text):
    #     return False

    return True

# -------------------------
def normalize_pdf_html(html_text: str) -> Tuple[str, dict]:
    soup = BeautifulSoup(html_text, "html.parser")
    num_pattern = re.compile(r'^(\d+(?:\.\d+)*\.?)([ \xa0]+|$)')

    app_tags = []
    for el in soup.find_all("h2"):
        if re.match(r'^Приложение\s+\S+', el.get_text(strip=True), re.I):
            app_tags.append(el)

    def in_appendix(el):
        if not el.sourceline:
            return False
        for i, app in enumerate(app_tags):
            start = app.sourceline
            end = app_tags[i + 1].sourceline if i + 1 < len(app_tags) else float("inf")
            if start < el.sourceline < end:
                return True
        return False

    for el in soup.find_all(["p", "li"] + [f"h{i}" for i in range(1, 7)]):
        text = el.get_text(strip=True).replace('\xa0', ' ')

        if re.match(r'^Приложение\s+\S+', text, re.I):
            continue

        # --- ВНУТРИ ПРИЛОЖЕНИЯ ---
        if in_appendix(el):
            if el.name in ["h2", "h3"]:
                level = int(el.name[1])
                el.name = f"h{min(level + 1, 6)}"
            continue

        # --- НУМЕРАЦИЯ ---
        m = num_pattern.match(text)
        if m:
            number = m.group(1).rstrip(".")
            if number.endswith('.'):
                number = number[:-1]
            rest = text[len(m.group(0)):].strip()

            level = min(number.count(".") + 2, 6)
            if level>2:
                if is_real_heading(rest):
                    h_tag = soup.new_tag(f"h{level}")
                    h_tag.string = f"{number} {rest}"
                    el.replace_with(h_tag)
                else:
                    h_tag = soup.new_tag(f"h{level}")
                    h_tag.string = number
                    el.replace_with(h_tag)

                    if rest:
                        p = soup.new_tag("p")
                        p.string = rest
                        h_tag.insert_after(p)
            else:
                    h_tag = soup.new_tag(f"h{level}")
                    h_tag.string = f"{number} {rest}"
                    el.replace_with(h_tag)

            continue

        # --- МАРКЕРЫ ---
        if re.match(r'^[\u2022\u25AA\u2023\-]\s*', text):
            el.string = re.sub(r'^[\u2022\u25AA\u2023\-]\s*', '', text)

    # --- ul / ol ---
    header_names = {f"h{i}" for i in range(1, 7)}
    for ul in soup.find_all(["ul", "ol"]):
        children = [c for c in ul.contents if isinstance(c, Tag)]
        if any(c.name in header_names for c in children):
            ul.unwrap()

    # --- Собираем секции ---
    sections = {}
    for h2 in soup.find_all("h2"):
        title = h2.get_text(strip=True)
        content = []
        for sib in h2.find_next_siblings():
            if sib.name == "h2":
                break
            content.append(str(sib))
        sections[title] = "\n".join(content)

    return str(soup), sections



# -------------------------
def wrap_with_div(html_text: str) -> str:
    soup = BeautifulSoup(html_text, "html.parser")
    
    # ИЗМЕНЕНИЕ: только h3, h2 НЕ трогаем!
    headers = soup.find_all("h3")  # ← БЫЛО ["h2", "h3"] → СТАЛО ["h3"]

    for header in headers:
        current_level = int(header.name[1])
        to_wrap = []
        sibling = header.find_next_sibling()

        while sibling:
            if isinstance(sibling, Tag):
                if sibling.name.startswith("h"):
                    if int(sibling.name[1]) <= current_level:
                        break
                if sibling.name in ["ul", "ol"]:
                    break
            to_wrap.append(sibling)
            sibling = sibling.next_sibling

        if to_wrap:
            div_tag = soup.new_tag("div", **{"class": "h-block"})
            for el in to_wrap:
                el.extract()
                div_tag.append(el)
            header.insert_after(div_tag)

    return str(soup)



# -------------------------
def remove_approval_blocks(html_text: str) -> str:
    soup = BeautifulSoup(html_text, "html.parser")
    approval_pattern = re.compile(
        r"утверждаю|ознакомлен[оа]?|ИНТЕГРИРОВАННАЯ СИСТЕМА МЕНЕДЖМЕНТА",
        re.I
    )

    to_remove = []

    for el in soup.find_all(["h2", "h3"]):
        if approval_pattern.search(el.get_text(strip=True)):
            to_remove.append(el)
            nxt = el.next_sibling
            while nxt and getattr(nxt, "name", None) not in ["h2", "h3"]:
                to_remove.append(nxt)
                nxt = nxt.next_sibling

    for el in to_remove:
        if hasattr(el, "extract"):
            el.extract()

    return str(soup)



# -------------------------
def clean_html_pipeline(html_text: str) -> str:
    """
    Полный пайплайн очистки HTML.
    """
    cleaned = remove_headers_footers_early(html_text)
    normalized, _ = normalize_pdf_html(cleaned)
    #wrapped = wrap_with_div(normalized)
    final = remove_approval_blocks(normalized)
    # html_dir = Path("/app/data/html")
    # html_path = html_dir / f"1.html"
    #html_path.write_text(final, encoding="utf-8")
    return final


#
