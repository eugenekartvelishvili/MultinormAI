from dataclasses import dataclass, field
from typing import List, Dict

# -------------------------
# Промежуточная секция после сплиттера
# -------------------------
@dataclass
class HtmlSection:
    doc_id: str
    section_id: int
    main_section_id: str
    title: str
    html: str
    metadata: Dict = field(default_factory=dict)


# -------------------------
# Итоговая секция после LLM
# -------------------------
@dataclass
class Subsection:
    subsection_id: str
    doc_id: str
    section_id: str
    main_section_id:str
    title: str         # заголовок H3
    number: str
    content: str       # текст H3 (не HTML)
    metadata: dict = field(default_factory=dict)


@dataclass
class Section:
    section_id: str
    main_section_id:str
    doc_id: str
    title: str
    number: str
    summary: str = ""   # краткое описание от LLM
    content: str = ""   # текст H2 до ближайшего H3
    metadata: dict = field(default_factory=dict)
    subsections: List[Subsection] = field(default_factory=list)


@dataclass
class Document:
    doc_id: str
    title: str
    sections: List[Section] = field(default_factory=list)
    summary: str = ""   # финальное summary всего документа
    metadata: dict = field(default_factory=dict)



# -------------------------
# Сырой документ из PDF/Word
# -------------------------
@dataclass
class RawDocument:
    doc_id: str
    html: str
    metadata: Dict = field(default_factory=dict)
