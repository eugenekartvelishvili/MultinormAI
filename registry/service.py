# code/registry/service.py
"""
Бизнес-логика реестра документов.
Архивация, версионирование, загрузка файлов.
Индексация вынесена в indexing_service.py.
"""

import logging
import re
from pathlib import Path
from typing import Optional

from pymilvus import Collection, utility, connections

from code.registry import db
from code.ingestion import config as cfg

logger = logging.getLogger(__name__)

UPLOADS_DIR = Path("/app/data/uploads")

COLLECTION_ACTUAL  = "doc_test_actual"
COLLECTION_ARCHIVE = "doc_test_archive"


def _safe_name(s: str) -> str:
    return re.sub(r"[^\w\-_.]", "_", s)


def _get_collection(name: str) -> Collection:
    col = Collection(name)
    col.load()
    return col


def _ensure_archive_collection():
    if utility.has_collection(COLLECTION_ARCHIVE):
        return
    src = Collection(COLLECTION_ACTUAL)
    schema = src.schema
    Collection(name=COLLECTION_ARCHIVE, schema=schema, using="default")
    for field in src.indexes:
        Collection(COLLECTION_ARCHIVE).create_index(
            field_name=field.field_name,
            index_params=field.params,
        )
    logger.info(f"Архивная коллекция {COLLECTION_ARCHIVE} создана")


def move_to_archive(milvus_doc_id: str):
    _ensure_archive_collection()
    src = _get_collection(COLLECTION_ACTUAL)
    dst = Collection(COLLECTION_ARCHIVE)
    results = src.query(
        expr=f'doc_id == "{milvus_doc_id}"',
        output_fields=["*"],
        limit=10000,
    )
    if not results:
        logger.warning(f"Чанков для doc_id={milvus_doc_id} не найдено")
        return
    dst.insert(results)
    dst.flush()
    src.delete(f'doc_id == "{milvus_doc_id}"')
    src.flush()
    logger.info(f"Скопировано {len(results)} чанков в архив")


def delete_from_milvus(milvus_doc_id: str):
    logger.info(f"[Milvus] Удаляем doc_id={milvus_doc_id}")
    try:
        col = _get_collection(COLLECTION_ACTUAL)
        result = col.delete(f'doc_id == "{milvus_doc_id}"')
        col.flush()
        logger.info(f"[Milvus] Удалено чанков: {result}")
    except Exception as e:
        logger.error(f"[Milvus] Ошибка удаления doc_id={milvus_doc_id}: {e}")
        raise


def archive_doc(registry_id: int):
    doc = db.get_by_id(registry_id)
    if not doc:
        raise ValueError(f"Документ #{registry_id} не найден")
    if doc["milvus_doc_id"]:
        move_to_archive(doc["milvus_doc_id"])
    db.archive_document(registry_id)


def archive_expired():
    expired = db.get_expired()
    if not expired:
        return
    for doc in expired:
        try:
            archive_doc(doc["id"])
        except Exception as e:
            logger.error(f"Ошибка архивации #{doc['id']}: {e}")


def prepare_upload(
    name: str,
    category: str,
    department: str,
    filename: str,
    author: str,
    valid_until: Optional[str],
    file_bytes: bytes,
    file_url: Optional[str] = None,
    revision: Optional[str] = None,
    revised_at: Optional[str] = None,
) -> tuple:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

    existing = db.get_by_name(name)
    new_version = 1
    if existing:
        archive_doc(existing["id"])
        new_version = existing["version"] + 1

    safe = _safe_name(Path(filename).stem)
    save_path = UPLOADS_DIR / f"{safe}_v{new_version}{Path(filename).suffix}"
    save_path.write_bytes(file_bytes)

    registry_id = db.create(
        name=name,
        category=category,
        department=department,
        filename=str(save_path),
        author=author,
        valid_until=valid_until or None,
        file_url=file_url,
        revision=revision or None,
        revised_at=revised_at or None,
        version=new_version,
    )
    return registry_id, save_path