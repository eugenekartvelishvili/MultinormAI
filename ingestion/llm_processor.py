# code/ingestion/llm_processor.py

import ast
import logging
import uuid
from typing import List
import gc
import torch
from bs4 import BeautifulSoup
from json_repair import repair_json
import json
from code.ingestion.models import HtmlSection, Section, Subsection, Document
from langchain_ollama import ChatOllama
from code.ingestion import config, prompts

logger = logging.getLogger(__name__)


class LLMProcessor:
    """
    Прогон HtmlSection через LLM и сбор готового Document.
    """

    def __init__(
        self,
        model: str = config.LLM_MODEL,
        base_url: str = config.LLM_BASE_URL,
        temperature: float = config.LLM_TEMPERATURE,
    ):
        self.llm = ChatOllama(model=model, temperature=temperature, base_url=base_url)

    # -------------------------------------------------
    # LLM вызов
    # -------------------------------------------------
    def unload_model(self):
        """Выгружаем модель ollama из GPU"""
        try:
            import httpx
            httpx.post(
                f"{config.LLM_BASE_URL}/api/generate",
                json={"model": config.LLM_MODEL, "keep_alive": 0},
                timeout=10
            )
            logger.info("💤 Ollama модель выгружена из GPU")
        except Exception as e:
            logger.warning(f"Не удалось выгрузить модель ollama: {e}")

    def _invoke_llm(
        self,
        section: HtmlSection,
        doc_name: str = "",
        category: str = "",
        department: str = "",
    ) -> dict:
        """
        Получаем словарь (Python dict) от LLM для одной HtmlSection.
        """
        import time

        payload = {
            "h2_title":   section.title or "",
            "html":       section.html,
            "doc_name":   doc_name   or "не указано",
            "category":   category   or "не указано",
            "department": department or "не указано",
        }
        user_prompt = prompts.USER_PROMPT_TEMPLATE.format(**payload)
        section_title = section.title or ""
        last_resp = ""

        for attempt in range(3):
            try:
                if attempt > 0:
                    gc.collect()
                    torch.cuda.empty_cache()
                    self.llm = ChatOllama(
                        model=config.LLM_MODEL,
                        temperature=config.LLM_TEMPERATURE,
                        base_url=config.LLM_BASE_URL,
                    )

                resp_content = self.llm.invoke([
                    {"role": "system", "content": prompts.SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ]).content
                resp_content = resp_content.strip()

                resp_content = resp_content.replace("\\", "\\\\")

                if resp_content.startswith("```"):
                    resp_content = resp_content.strip("`")
                    resp_content = resp_content.replace("json", "", 1).strip()

                last_resp = resp_content

                try:
                    result = json.loads(resp_content)
                    if isinstance(result, dict):
                        return result
                except Exception:
                    pass

                try:
                    fixed = repair_json(resp_content)
                    result = json.loads(fixed)
                    if isinstance(result, dict):
                        return result
                except Exception:
                    pass

                try:
                    import re
                    safe = re.sub(r'\n\s*",\s*\n', '\n', resp_content)
                    safe = safe.replace("'", '"')
                    result = json.loads(safe)
                    if isinstance(result, dict):
                        return result
                except Exception:
                    pass

                raise ValueError("Невозможно преобразовать ответ LLM в dict")

            except Exception as e:
                logger.warning(f"[Попытка {attempt + 1}] Ошибка LLM для секции '{section_title}': {e}")
                time.sleep(1)

        logger.error(f"❌ Все попытки получить валидный dict для секции '{section_title}' завершились неудачей")
        logger.error(f"Последний ответ LLM:\n{last_resp}")
        return None

    # -------------------------------------------------
    # Парсинг LLM dict
    # -------------------------------------------------

    def _parse_llm_json(
        self,
        *,
        doc_id: str,
        llm_dict: dict,
        html_section: HtmlSection,
    ) -> Section:
        """
        Преобразуем ответ LLM (уже Python dict) в Section / Subsection.
        """
        section_id = str(uuid.uuid4())
        main_section_id = html_section.main_section_id or section_id

        title   = llm_dict.get("h2_title") or html_section.title or "Без заголовка"
        number  = llm_dict.get("number")   or html_section.metadata.get("number", "")
        content = BeautifulSoup(llm_dict.get("content") or "", "html.parser").get_text("\n", strip=True)
        summary = BeautifulSoup(llm_dict.get("summary") or "", "html.parser").get_text("\n", strip=True)

        chunks = llm_dict.get("chunks") or []
        subsections: list[Subsection] = []

        def extract_tables_markdown(tables) -> str:
            parts = []
            if isinstance(tables, list):
                for t in tables:
                    if isinstance(t, dict) and "markdown" in t:
                        parts.append(t["markdown"])
                    elif isinstance(t, str):
                        parts.append(t)
            elif isinstance(tables, str):
                parts.append(tables)
            return "\n\n".join(parts)

        for chunk in chunks:
            if not isinstance(chunk, dict) or chunk.get("type") != "h3":
                continue

            chunk_text = BeautifulSoup(chunk.get("content") or "", "html.parser").get_text("\n", strip=True)
            tables_md  = extract_tables_markdown(chunk.get("tables", []))

            subsections.append(
                Subsection(
                    subsection_id=str(uuid.uuid4()),
                    doc_id=doc_id,
                    section_id=section_id,
                    main_section_id=main_section_id,
                    title=chunk.get("title") or "",
                    number=chunk.get("number") or "",
                    content=chunk_text,
                    metadata={"tables": chunk.get("tables", []), "tables_md": tables_md},
                )
            )

        return Section(
            section_id=section_id,
            main_section_id=main_section_id,
            doc_id=doc_id,
            title=title,
            number=number,
            summary=summary,
            content=content,
            subsections=subsections,
            metadata={
                "section_index": html_section.metadata.get("section_index"),
                "chunk_index":   html_section.metadata.get("chunk_index"),
            },
        )

    # -------------------------------------------------
    # Прогон HtmlSection через LLM
    # -------------------------------------------------

    def process_html_sections(
        self,
        html_sections: List[HtmlSection],
        doc_title: str,
        doc_name: str = "",
        category: str = "",
        department: str = "",
        progress_callback=None,
    ) -> Document:
        """
        Преобразуем HtmlSection через LLM и формируем Document.
        """
        doc_id = str(uuid.uuid4())
        sections: List[Section] = []

        for sec in html_sections:
            if not sec.html.strip():
                logger.info(f"Пропускаем пустой блок H2: {sec.title}")
                continue

            llm_dict = self._invoke_llm(
                sec,
                doc_name=doc_name,
                category=category,
                department=department,
            )
            if not llm_dict:
                logger.error(f"⛔ Секция '{sec.title}' пропущена из-за ошибки LLM")
                continue

            section = self._parse_llm_json(
                doc_id=doc_id,
                llm_dict=llm_dict,
                html_section=sec,
            )

            sections.append(section)

            if progress_callback:
                progress_callback()

            logger.info(
                f"✅ Обработан H2: {section.number} {section.title} "
                f"(main_section_id={section.main_section_id})"
            )

        # --- summary документа ---
        section_summaries = [f"- {s.summary}" for s in sections if s.summary]
        doc_summary = ""

        if section_summaries:
            user_prompt = prompts.USER_PROMPT_DOC.format(
                doc_name=doc_name or doc_title,
                category=category or "не указано",
                department=department or "не указано",
                section_summaries="\n".join(section_summaries)
            )
            try:
                resp = self.llm.invoke([
                    {"role": "system", "content": prompts.SYSTEM_PROMPT_DOC},
                    {"role": "user",   "content": user_prompt},
                ])
                doc_summary = resp.content.strip()
                logger.info("✅ Сгенерировано summary документа")
            except Exception:
                logger.exception("⚠ Ошибка генерации summary документа")

        return Document(
            doc_id=doc_id,
            title=doc_title,
            sections=sections,
            summary=doc_summary,
            metadata={},
        )