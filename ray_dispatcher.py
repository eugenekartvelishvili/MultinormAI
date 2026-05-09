# code/ray_dispatcher.py
import asyncio
import logging
import sys
import os

import ray

logger = logging.getLogger(__name__)

RAY_ADDRESS = "ray://ray-head:10001"


@ray.remote(num_gpus=1)
class RAGActor:
    def __init__(self, actor_id: int, ollama_url: str):
        if "/app" not in sys.path:
            sys.path.insert(0, "/app")

        import logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(f"RAGActor.{actor_id}")
        self.actor_id = actor_id
        self.ollama_url = ollama_url

        self.logger.info(f"[RAGActor {actor_id}] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

        from code.ingestion.embedding import EmbeddingService
        from code.retrieval.rag.milvus_client import MilvusApiClient
        from pymilvus import connections
        from code.ingestion import config as cfg

        connections.connect(host=cfg.MILVUS_HOST, port=cfg.MILVUS_PORT)

        # EmbeddingService инициализируется на CPU — GPU не занимаем до запроса
        self._embedder = EmbeddingService()
        self._client = MilvusApiClient(embedder=self._embedder)
        self.logger.info(f"[RAGActor {actor_id}] готов, ollama={ollama_url}")

    def process(self, question: str) -> dict:
        if "/app" not in sys.path:
            sys.path.insert(0, "/app")

        from langchain_ollama import ChatOllama
        from code.ingestion import config as cfg
        from code.retrieval.rag.classifier import classify_question
        from code.retrieval.rag.rewriter import rewrite_query
        from code.retrieval.rag.answer_generator import generate_answer
        from code.retrieval.rag.strategies import strategy_a, strategy_b, strategy_c
        from code.retrieval.rag.context import build_context, build_sources, format_sources_bbcode
        from code.retrieval.rag.router import markdown_to_bbcode
        import time

        llm = ChatOllama(model=cfg.LLM_MODEL, temperature=0.0, base_url=self.ollama_url)
        llm_answer = ChatOllama(model=cfg.LLM_MODEL_ANSWER, temperature=0.0, base_url=self.ollama_url)

        t0 = time.time()
        self.logger.info(f"[RAGActor {self.actor_id}] Вопрос: {question!r}")

        classification   = classify_question(question, llm=llm)
        question_type    = classification["question_type"]
        answer_mode      = classification["answer_mode"]
        needs_decomp     = classification["needs_decomposition"]

        rewrite          = rewrite_query(question, needs_decomp, llm=llm)
        rewritten        = rewrite["rewritten"]
        important_tokens = rewrite["important_tokens"]
        entity           = rewrite.get("entity", rewritten)
        attribute        = rewrite.get("attribute", rewritten)

        # Грузим эмбеддер на GPU только на время поиска
        self._embedder.load()
        try:
            if question_type == "norm_reference":
                blocks = strategy_c(self._client, rewritten, important_tokens)
            elif question_type in ("procedure", "broad_overview"):
                blocks = strategy_b(self._client, entity, attribute, important_tokens)
            else:
                blocks = strategy_a(self._client, rewritten, important_tokens)
        finally:
            self._embedder.unload()

        context = build_context(blocks)
        sources = build_sources(blocks, max_sources=3)

        if context.strip():
            answer = generate_answer(
                question=question, context=context,
                question_type=question_type, answer_mode=answer_mode,
                llm=llm_answer,
            )
        else:
            answer = "В документах нет информации по этому вопросу."

        answer_bbcode  = markdown_to_bbcode(answer.strip())
        answer_bbcode += "\n\n" + format_sources_bbcode(sources)

        self.logger.info(f"[RAGActor {self.actor_id}] Готово за {time.time()-t0:.2f}s")

        return {
            "question":       question,
            "question_type":  question_type,
            "answer_mode":    answer_mode,
            "classification": classification,
            "results":        [{"source": b.source, "score": b.score, "number": b.number} for b in blocks],
            "context":        context,
            "answer":         answer_bbcode,
            "sources":        sources,
        }


@ray.remote(num_gpus=1)
class IndexingActor:
    """Актор для индексации — запускается по требованию, освобождает GPU после."""

    def __init__(self):
        if "/app" not in sys.path:
            sys.path.insert(0, "/app")
        import logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("IndexingActor")
        self.logger.info(f"[IndexingActor] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    def index(self, registry_id: int, file_path: str, department: str,
              category: str, doc_name: str) -> dict:
        if "/app" not in sys.path:
            sys.path.insert(0, "/app")

        import gc
        import torch
        from pathlib import Path
        from code.registry import db
        from code.ingestion import config as cfg
        from pymilvus import Collection, connections

        connections.connect(host=cfg.MILVUS_HOST, port=cfg.MILVUS_PORT)

        db.update_fields(registry_id, index_status="indexing")
        self.logger.info(f"[IndexingActor] Индексация #{registry_id}: {file_path}")

        try:
            from code.ingestion.document_reader import DocumentReader
            from code.ingestion.llm_processor import LLMProcessor
            from code.ingestion.document_processor import DocumentProcessor
            from code.ingestion.milvus_indexer import MilvusIndexer
            from code.ingestion.embedding import EmbeddingService

            reader = DocumentReader(artifacts_path=cfg.ARTIFACTS_PATH)
            llm_processor = LLMProcessor()
            processor = DocumentProcessor(reader=reader, llm_processor=llm_processor)

            # process() сам вызывает load/unload внутри через embedder
            document = processor.process(
                Path(file_path),
                doc_name=doc_name,
                category=category,
                department=department,
            )

            import json
            from dataclasses import asdict
            import re
            from pathlib import Path as P
            json_dir = P("/app/data/json")
            json_dir.mkdir(parents=True, exist_ok=True)
            safe = re.sub(r"[^\w\-_.]", "_", P(file_path).stem)
            with open(json_dir / f"{safe}.json", "w", encoding="utf-8") as f:
                json.dump(asdict(document), f, ensure_ascii=False, indent=2)

            db.update_fields(registry_id, index_status="indexed",
                             milvus_doc_id=document.doc_id, html_path=None)
            self.logger.info(f"[IndexingActor] #{registry_id} проиндексирован")
            return {"ok": True, "doc_id": document.doc_id}

        except Exception as e:
            self.logger.error(f"[IndexingActor] Ошибка #{registry_id}: {e}")
            db.update_fields(registry_id, index_status="error", error_msg=str(e)[:500])
            return {"ok": False, "error": str(e)}


class RayDispatcher:
    def __init__(self):
        from code.ingestion import config as cfg

        if not ray.is_initialized():
            ray.init(RAY_ADDRESS, ignore_reinit_error=True)

        runtime_env = {"env_vars": {"PYTHONPATH": "/app"}}

        # GPU 0 — постоянный актор для ответов
        self._permanent_actor = RAGActor.options(
            runtime_env=runtime_env
        ).remote(actor_id=0, ollama_url=cfg.LLM_URLS[0])

        self._ollama_urls = cfg.LLM_URLS
        self._runtime_env = runtime_env

        # GPU 1 — для временного RAGАктора или индексации
        self._temp_actor     = None
        self._indexing_actor = None
        self._gpu1_lock      = asyncio.Lock()

        logger.info("[RayDispatcher] запущен, постоянный актор на GPU 0")

    async def process(self, question: str) -> dict:
        result = await self._permanent_actor.process.remote(question)
        return result

    async def process_parallel(self, question: str) -> dict:
        async with self._gpu1_lock:
            if self._temp_actor is None:
                self._temp_actor = RAGActor.options(
                    runtime_env=self._runtime_env
                ).remote(actor_id=1, ollama_url=self._ollama_urls[1])
            result = await self._temp_actor.process.remote(question)
        return result

    async def index(self, registry_id: int, file_path: str, department: str,
                    category: str, doc_name: str):
        """
        Запускает индексацию на GPU 1.
        Если GPU 1 занята временным RAGАктором — убивает его и ждёт.
        """
        async with self._gpu1_lock:
            # Освобождаем GPU 1 если там был временный RAGАктор
            if self._temp_actor is not None:
                ray.kill(self._temp_actor)
                self._temp_actor = None

            actor = IndexingActor.options(
                runtime_env=self._runtime_env
            ).remote()

            try:
                result = await actor.index.remote(
                    registry_id, file_path, department, category, doc_name
                )
            finally:
                ray.kill(actor)
                self._indexing_actor = None

            return result