# code/registry/indexing_service.py
"""
Запускает индексацию через RayDispatcher если доступен,
иначе fallback на прямую индексацию.
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


async def run_indexing(
    registry_id: int,
    file_path: Path,
    department: str,
    category: str = "",
    doc_name: str = "",
):
    from code.api.main import dispatcher

    if dispatcher is not None:
        logger.info(f"[indexing] #{registry_id} через RayDispatcher")
        await dispatcher.index(
            registry_id=registry_id,
            file_path=str(file_path),
            department=department,
            category=category,
            doc_name=doc_name,
        )
    else:
        logger.info(f"[indexing] #{registry_id} fallback (Ray недоступен)")
        await _fallback_indexing(registry_id, file_path, department, category, doc_name)


async def _fallback_indexing(
    registry_id: int,
    file_path: Path,
    department: str,
    category: str = "",
    doc_name: str = "",
):
    """Прямая индексация без Ray — как было раньше."""
    import gc
    import urllib.request
    import json as _json
    from pymilvus import connections
    from code.ingestion import config as cfg
    from code.registry import db

    async def _ollama_unload():
        try:
            data = _json.dumps({"model": cfg.LLM_MODEL, "keep_alive": 0}).encode()
            req = urllib.request.Request(
                f"{cfg.LLM_BASE_URL}/api/generate",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            urllib.request.urlopen(req, timeout=10)
        except Exception as e:
            logger.warning(f"Не удалось выгрузить Ollama: {e}")

    db.update_fields(registry_id, index_status="indexing")
    try:
        await _ollama_unload()
        connections.connect(alias="default", host=cfg.MILVUS_HOST, port=cfg.MILVUS_PORT)
        loop = asyncio.get_event_loop()
        doc = await loop.run_in_executor(
            None, _process_sync,
            file_path, department, category, doc_name,
        )
        db.update_fields(registry_id, index_status="indexed",
                        milvus_doc_id=doc.doc_id, html_path=None)
        logger.info(f"Документ #{registry_id} проиндексирован")
    except Exception as e:
        logger.exception(f"Ошибка индексации #{registry_id}")
        db.update_fields(registry_id, index_status="error", error_msg=str(e)[:500])


def _process_sync(file_path, department, category, doc_name):
    import gc
    from pymilvus import Collection
    from code.ingestion.document_reader import DocumentReader
    from code.ingestion.llm_processor import LLMProcessor
    from code.ingestion.document_processor import DocumentProcessor
    from code.ingestion.milvus_indexer import MilvusIndexer
    from code.ingestion.embedding import EmbeddingService
    from code.ingestion import config as cfg

    reader    = DocumentReader(artifacts_path=cfg.ARTIFACTS_PATH)
    processor = DocumentProcessor(reader=reader, llm_processor=LLMProcessor())

    # Метаданные прокидываются внутрь processor.process — дублировать не нужно
    document = processor.process(
        Path(file_path),
        doc_name=doc_name,
        category=category,
        department=department,
    )

    import json
    import re
    from pathlib import Path as _Path
    from dataclasses import asdict
    json_dir = _Path("/app/data/json")
    json_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^\w\-_.]", "_", _Path(str(file_path)).stem)
    with open(json_dir / f"{safe}.json", "w", encoding="utf-8") as f:
        json.dump(asdict(document), f, ensure_ascii=False, indent=2)

    return document