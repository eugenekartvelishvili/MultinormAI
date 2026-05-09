# ingestion/milvus_indexer.py

import logging
import re
from typing import List

from pymilvus import Collection
from code.ingestion.models import Document, Section, Subsection
from code.ingestion.embedding import EmbeddingService
from code.ingestion import config

logger = logging.getLogger(__name__)

EMPTY_DENSE = [0.0] * config.EMBED_DENSE_DIM


def _clean_version(title: str) -> str:
    """Убирает технические суффиксы версий: _v1, _v2, _v1.1 и т.д."""
    return re.sub(r'_v\d+(\.\d+)?$', '', title, flags=re.IGNORECASE).strip()


class MilvusIndexer:
    def __init__(self, collection: Collection, embedder: EmbeddingService):
        self.collection = collection
        self.embedder = embedder

    def index_document(self, document: Document):
        logger.info(f"Индексация документа: {document.title}")

        department = getattr(document, "_department", "")
        category   = getattr(document, "_category",   "")
        doc_name   = getattr(document, "_doc_name",   document.title)

        # Человеческое название без суффикса _v1
        human_title = _clean_version(doc_name.strip()) if doc_name.strip() else _clean_version(document.title)

        records: List[dict] = []

        doc_record = self._doc_record(document, department, category, human_title)
        if doc_record:
            records.append(doc_record)

        for section in document.sections:
            section_record = self._section_record(document, section, department, category, human_title)
            if section_record:
                records.append(section_record)

            for subsection in section.subsections:
                subsection_record = self._subsection_record(document, section, subsection, department, category, human_title)
                if subsection_record:
                    records.append(subsection_record)

        if not records:
            logger.warning("Нет записей для вставки")
            return

        self._insert(records)
        logger.info(f"Вставлено записей: {len(records)}")

    def _doc_record(self, document: Document, department: str, category: str, human_title: str) -> dict:
        """
        level=0 запись — представляет весь документ.
        text и summary строятся с явным включением category + human_title + department,
        чтобы при поиске entity="отдел геологии" Положение побеждало должностные инструкции.
        """
        tech_title = document.title.strip()  # "ПЛ-07_v1" — для внутренней ссылки

        # text: категория + название + отдел + заголовок первого раздела
        # именно этот текст эмбеддируется в dense_vector (content)
        text_parts = [p for p in [category, human_title, department] if p]
        if document.sections and document.sections[0].title:
            text_parts.append(document.sections[0].title.strip())
        text = "\n".join(text_parts)

        # summary: берём из первой секции или document.summary,
        # префиксируем контекстом документа чтобы summary-вектор тоже знал о чём документ
        raw_summary = ""
        if document.sections and document.sections[0].summary:
            raw_summary = document.sections[0].summary.strip()
        if not raw_summary:
            raw_summary = document.summary or ""

        if raw_summary:
            prefix_parts = [p for p in [category, human_title, department] if p]
            doc_summary = " / ".join(prefix_parts) + ". " + raw_summary
        else:
            doc_summary = " / ".join([p for p in [category, human_title, department] if p])

        content_emb = self.embedder([text])
        summary_emb = self.embedder([doc_summary]) if doc_summary else None

        return {
            "id":                   document.doc_id,
            "text":                 text,
            "dense_vector":         content_emb["dense"][0].tolist(),
            "sparse_vector":        content_emb["sparse"],
            "dense_vector_summary": summary_emb["dense"][0].tolist() if summary_emb else EMPTY_DENSE,
            "level":                0,
            "doc_id":               document.doc_id,
            "title":                human_title,              # человеческое название — для отображения
            "doc_title":            _clean_version(tech_title),  # технический ID без _v1
            "doc_name":             human_title,
            "department":           department,
            "category":             category,
        }

    def _section_record(self, document: Document, section: Section, department: str, category: str, human_title: str) -> dict:
        tables_md = ""
        if section.metadata and section.metadata.get("tables_md"):
            tables_md = f"\n[ТАБЛИЦА]\n{section.metadata['tables_md']}"

        content_text = f"{section.title}\n{section.content}{tables_md}".strip()
        summary_text = section.summary.strip() if section.summary else ""

        if not content_text and not summary_text:
            return None

        content_emb = self.embedder([content_text])
        summary_emb = self.embedder([summary_text]) if summary_text else None

        return {
            "id":                   section.section_id,
            "text":                 content_text,
            "dense_vector":         content_emb["dense"][0].tolist(),
            "sparse_vector":        content_emb["sparse"],
            "dense_vector_summary": summary_emb["dense"][0].tolist() if summary_emb else EMPTY_DENSE,
            "level":                1,
            "doc_id":               document.doc_id,
            "doc_title":            human_title,          # человеческое, без _v1
            "section_id":           section.section_id,
            "main_section_id":      section.main_section_id,
            "number":               section.number,
            "title":                section.title,
            "doc_name":             human_title,
            "department":           department,
            "category":             category,
        }

    def _subsection_record(self, document: Document, section: Section, subsection: Subsection, department: str, category: str, human_title: str) -> dict:
        tables_md = ""
        if subsection.metadata and subsection.metadata.get("tables_md"):
            tables_md = f"\n[ТАБЛИЦА]\n{subsection.metadata['tables_md']}"

        text = f"{subsection.title}\n{subsection.content}{tables_md}".strip()

        if not text:
            return None

        emb = self.embedder([text])

        return {
            "id":                   subsection.subsection_id,
            "text":                 text,
            "dense_vector":         emb["dense"][0].tolist(),
            "sparse_vector":        emb["sparse"],
            "dense_vector_summary": EMPTY_DENSE,
            "level":                2,
            "doc_id":               document.doc_id,
            "doc_title":            human_title,          # человеческое, без _v1
            "section_title":        section.title,
            "section_id":           section.section_id,
            "main_section_id":      subsection.main_section_id,
            "subsection_id":        subsection.subsection_id,
            "number":               subsection.number,
            "title":                subsection.title,
            "doc_name":             human_title,
            "department":           department,
            "category":             category,
        }

    def _insert(self, records: List[dict]):
        if not records:
            return
        self.collection.insert(records)
        self.collection.flush()
        logger.info("Данные записаны в Milvus")