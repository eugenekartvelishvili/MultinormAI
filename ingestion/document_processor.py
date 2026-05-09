# code/ingestion/document_processor.py

from pathlib import Path
import logging
import re
from typing import List
from rich.logging import RichHandler
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
)

from code.ingestion.document_reader import DocumentReader
from code.ingestion.cleaner import clean_html_pipeline
from code.ingestion.splitter import split_html_by_h2
from code.ingestion.models import HtmlSection, Document
from code.ingestion.llm_processor import LLMProcessor
from code.ingestion.milvus_client import get_collection
from code.ingestion.milvus_indexer import MilvusIndexer
from code.ingestion.embedding import EmbeddingService

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Полный пайплайн обработки одного документа:
    файл → HTML → очистка → HtmlSection → LLM → Milvus
    Используется только для CLI запуска, не в продакшене.
    В продакшене индексация идёт через service.py.
    """

    def __init__(self, reader: DocumentReader, llm_processor: LLMProcessor):
        self.reader = reader
        self.llm_processor = llm_processor

    def process(
        self,
        path: Path,
        doc_name: str = "",
        category: str = "",
        department: str = "",
    ) -> Document:

        with Progress(
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(),
            TextColumn("{task.percentage:>3.0f}%"),
            TextColumn("•"),
            TextColumn("[dim]{task.fields[stage]}"),
            TimeElapsedColumn(),
            transient=False,
            refresh_per_second=4,
        ) as progress:

            main_task = progress.add_task(
                description=path.name,
                total=None,
                stage="Чтение документа...",
            )

            # 1. Чтение
            raw = self.reader.read(path)
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            progress.update(main_task, stage="Подготовка секций...")

            # 2. Очистка + Split
            clean_html = clean_html_pipeline(raw.html)

            html_dir = Path("/app/data/html")
            html_dir.mkdir(parents=True, exist_ok=True)
            safe_name = re.sub(r'[^\w\-_. ]', '_', path.stem)
            html_path = html_dir / f"{safe_name}.html"
            try:
                html_path.write_text(clean_html, encoding="utf-8")
                logger.info(f"[HTML] сохранён: {html_path}")
            except Exception as e:
                logger.warning(f"[HTML] ошибка сохранения {html_path}: {e}")

            html_sections: List[HtmlSection] = split_html_by_h2(
                doc_id=raw.doc_id,
                html=clean_html,
                base_metadata=raw.metadata,
            )
            n = len(html_sections)

            progress.update(main_task, total=n + 3, completed=2,
                            stage=f"LLM (0/{n})")

            llm_task = progress.add_task(
                description="  🤖 Чанки",
                total=n,
                stage="...",
            )

            # 3. LLM — передаём контекст документа
            processed = [0]

            def on_chunk_done():
                processed[0] += 1
                progress.update(llm_task, advance=1,
                                stage=f"{processed[0]}/{n}")
                progress.update(main_task, advance=1,
                                stage=f"LLM ({processed[0]}/{n})")

            document = self.llm_processor.process_html_sections(
                html_sections=html_sections,
                doc_title=path.stem,
                doc_name=doc_name,
                category=category,
                department=department,
                progress_callback=on_chunk_done,
            )

            progress.remove_task(llm_task)

            self.llm_processor.unload_model()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 4. Индексация — прокидываем метаданные
            progress.update(main_task, advance=1, stage="Загрузка в Milvus...")
            document._department = department
            document._category   = category
            document._doc_name   = doc_name or document.title

            collection = get_collection()
            embedder = EmbeddingService()
            embedder.load()
            try:
                indexer = MilvusIndexer(collection=collection, embedder=embedder)
                indexer.index_document(document)
            finally:
                embedder.unload()
                import gc; gc.collect()

            progress.update(main_task, advance=1, stage="Готово ✅")

            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return document