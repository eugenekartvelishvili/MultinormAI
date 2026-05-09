# code/api/documents_router.py
"""
Роутер реестра документов.
"""

import logging
from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional

from code.registry import db as registry_db
from code.registry import service as registry_svc
from code.registry.indexing_service import run_indexing

logger = logging.getLogger(__name__)

router = APIRouter()


class StatusUpdate(BaseModel):
    status: str


class DocumentEdit(BaseModel):
    name:        Optional[str] = None
    category:    Optional[str] = None
    department:  Optional[str] = None
    valid_until: Optional[str] = None
    revision:    Optional[str] = None
    revised_at:  Optional[str] = None
    author:      Optional[str] = None


@router.get("")
def list_documents(
    status: Optional[str] = None,
    index_status: Optional[str] = None,
    category: Optional[str] = None,
    department: Optional[str] = None,
    search: Optional[str] = None,
):
    docs = registry_db.get_all(
        status=status, index_status=index_status,
        category=category, department=department, search=search,
    )
    return {"ok": True, "data": docs}


@router.get("/{doc_id}/index-status")
def get_index_status(doc_id: int):
    doc = registry_db.get_by_id(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Документ не найден")
    return {"ok": True, "id": doc_id, "index_status": doc["index_status"], "error_msg": doc.get("error_msg")}


@router.get("/{doc_id}")
def get_document(doc_id: int):
    doc = registry_db.get_by_id(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Документ не найден")
    return {"ok": True, "data": doc}


@router.post("")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    name: str = Form(...),
    category: str = Form("Прочее"),
    department: str = Form("ВолгоградНИПИнефть"),
    valid_until: Optional[str] = Form(None),
    author: str = Form("API"),
    file_url: Optional[str] = Form(None),
    replace_id: Optional[int] = Form(None),
    revision: Optional[str] = Form(None),
    revised_at: Optional[str] = Form(None),
):
    try:
        file_bytes = await file.read()

        if replace_id:
            old_doc = registry_db.get_by_id(replace_id)
            if old_doc and old_doc["status"] == "active":
                registry_svc.archive_doc(replace_id)

        registry_id, save_path = registry_svc.prepare_upload(
            name=name,
            category=category,
            department=department,
            filename=file.filename,
            author=author,
            file_url=file_url,
            valid_until=valid_until,
            file_bytes=file_bytes,
            revision=revision or None,
            revised_at=revised_at or None,
        )

        background_tasks.add_task(
            run_indexing,
            registry_id, save_path, department, category, name,
        )

        return {"ok": True, "data": registry_db.get_by_id(registry_id)}

    except Exception as e:
        logger.exception("Ошибка загрузки документа")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/{doc_id}/status")
def set_status(doc_id: int, body: StatusUpdate):
    doc = registry_db.get_by_id(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Документ не найден")
    if body.status not in {"active", "archived"}:
        raise HTTPException(status_code=400, detail="status: active | archived")
    if body.status == "archived" and doc["status"] != "archived":
        try:
            registry_svc.archive_doc(doc_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    else:
        registry_db.update_fields(doc_id, status=body.status)
    return {"ok": True, "data": registry_db.get_by_id(doc_id)}


@router.patch("/{doc_id}")
def edit_document(doc_id: int, body: DocumentEdit):
    doc = registry_db.get_by_id(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Документ не найден")

    updates = {k: v for k, v in body.dict().items() if v is not None}
    if not updates:
        return {"ok": True, "data": doc}

    registry_db.update_fields(doc_id, **updates)

    if doc.get("milvus_doc_id") and doc.get("index_status") == "indexed":
        try:
            _update_milvus_metadata(
                milvus_doc_id=doc["milvus_doc_id"],
                name=body.name or doc["name"],
                category=body.category or doc["category"],
                department=body.department or doc["department"],
            )
        except Exception as e:
            logger.warning(f"Не удалось обновить метаданные в Milvus: {e}")

    return {"ok": True, "data": registry_db.get_by_id(doc_id)}


def _update_milvus_metadata(milvus_doc_id: str, name: str, category: str, department: str):
    from pymilvus import Collection
    from code.registry.service import COLLECTION_ACTUAL
    col = Collection(COLLECTION_ACTUAL)
    col.load()
    results = col.query(
        expr=f'doc_id == "{milvus_doc_id}"',
        output_fields=["*"],
        limit=10000,
    )
    if not results:
        return
    for r in results:
        r["doc_name"]   = name
        r["category"]   = category
        r["department"] = department
    col.delete(f'doc_id == "{milvus_doc_id}"')
    col.flush()
    col.insert(results)
    col.flush()
    logger.info(f"Обновлены метаданные Milvus: doc_id={milvus_doc_id}, {len(results)} чанков")


@router.delete("/{doc_id}")
def delete_document(doc_id: int):
    doc = registry_db.get_by_id(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Документ не найден")
    if doc.get("milvus_doc_id"):
        try:
            registry_svc.delete_from_milvus(doc["milvus_doc_id"])
        except Exception as e:
            logger.warning(f"Не удалось удалить из Milvus: {e}")
    registry_db.delete(doc_id)
    return {"ok": True, "deleted_id": doc_id}


@router.post("/archive-expired")
def archive_expired():
    registry_svc.archive_expired()
    return {"ok": True}