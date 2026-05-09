# code/api/main.py

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Optional, List
from collections import defaultdict
import logging
import httpx
import time
import asyncio
import json

from code.retrieval.rag.router import adaptive_search
from code.api.documents_router import router as documents_router
from code.registry import db as registry_db

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Multinorm Milvus API")
app.include_router(documents_router, prefix="/documents", tags=["documents"])

session_histories = defaultdict(list)
dispatcher = None


class AskRequest(BaseModel):
    text: str
    session_id: str = "default"


@app.on_event("startup")
async def startup():
    global dispatcher
    registry_db.init_db()

    try:
        from code.retrieval.rag.milvus_client import MilvusApiClient
        MilvusApiClient()
        logging.info("✅ MilvusApiClient инициализирован")
    except Exception as e:
        logging.error(f"❌ Ошибка инициализации MilvusApiClient: {e}")

    try:
        import ray
        ray.init("ray://ray-head:10001", ignore_reinit_error=True)
        from code.ray_dispatcher import RayDispatcher
        dispatcher = RayDispatcher()
        logging.info("✅ RayDispatcher инициализирован")
    except Exception as e:
        logging.warning(f"⚠ Ray недоступен, fallback на gpu_lock: {e}")
        dispatcher = None


async def _process_question(question: str) -> dict:
    if dispatcher is not None:
        return await dispatcher.process(question)
    from code.registry.gpu_lock import gpu_lock
    async with gpu_lock:
        return await asyncio.get_event_loop().run_in_executor(
            None, adaptive_search, question
        )


@app.get("/ping")
def ping():
    return {"status": "ok", "ray": dispatcher is not None}


@app.post("/ask")
async def ask(req: AskRequest):
    try:
        history = session_histories[req.session_id]
        _t = time.perf_counter()
        result = await _process_question(req.text)
        elapsed = time.perf_counter() - _t
        logging.info(f"[TIME] /ask total={elapsed:.2f}s")
        history.append({
            "question": req.text,
            "answer": result["answer"],
            "question_type": result["question_type"],
        })
        if len(history) > 10:
            session_histories[req.session_id] = history[-10:]
        return {
            "answer": result["answer"],
            "question_type": result["question_type"],
            "answer_mode": result["answer_mode"],
            "sources": result["results"][:3],
        }
    except Exception as e:
        logging.exception("Ошибка при /ask")
        raise HTTPException(status_code=500, detail=str(e))


BITRIX_WEBHOOK = "https://bitrix.volgogradnipineft.com/rest/136/rod18t5uf3ap0d5w/"
BITRIX_BOT_ID = "599"
BITRIX_CLIENT_ID = "ctapegsz9g0ugv3dgg2hweof2ji2l3q9"


@app.post("/bitrix")
async def bitrix_webhook(request: Request):
    try:
        data = await request.json()
        logging.info(f"Bitrix full data: {json.dumps(data, ensure_ascii=False)}")

        event = data.get("event")
        if event != "ONIMBOTMESSAGEADD":
            return {"status": "ok"}

        user_id = str(data.get("data", {}).get("USER", {}).get("ID", "unknown"))
        message_text = data.get("data", {}).get("PARAMS", {}).get("MESSAGE", "")
        dialog_id = data.get("data", {}).get("PARAMS", {}).get("DIALOG_ID", "")

        if not message_text:
            return {"status": "ok"}

        history = session_histories[user_id]
        _t = time.perf_counter()
        result = await _process_question(message_text)
        elapsed = time.perf_counter() - _t
        logging.info(f"[TIME] /bitrix rag={elapsed:.2f}s")

        history.append({
            "question": message_text,
            "answer": result["answer"],
            "question_type": result["question_type"],
        })
        if len(history) > 10:
            session_histories[user_id] = history[-10:]

        async with httpx.AsyncClient(verify=False) as client:
            resp = await client.post(
                f"{BITRIX_WEBHOOK}imbot.message.add.json",
                json={
                    "BOT_ID": BITRIX_BOT_ID,
                    "CLIENT_ID": BITRIX_CLIENT_ID,
                    "DIALOG_ID": dialog_id,
                    "MESSAGE": result["answer"],
                },
                timeout=30,
            )
            logging.info(f"Bitrix response: {resp.status_code} {resp.text}")

        logging.info(f"[TIME] /bitrix total={time.perf_counter()-_t:.2f}s")
        return {"status": "ok"}

    except Exception as e:
        logging.exception("Ошибка в bitrix webhook")
        return {"status": "error", "detail": str(e)}