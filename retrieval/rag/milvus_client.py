# milvus_client.py  (code/retrieval/rag/milvus_client.py)
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from pymilvus import Collection, connections

from code.retrieval.rag.config import MILVUS_HOST, MILVUS_PORT, MILVUS_COLLECTION
from code.ingestion.embedding import EmbeddingService
from code.retrieval.query import SearchQuery
from code.retrieval.searcher import MilvusSearcher

logger = logging.getLogger(__name__)

_instance: Optional["MilvusApiClient"] = None


class MilvusApiClient:

    def __new__(cls, embedder: Optional[EmbeddingService] = None) -> "MilvusApiClient":
        global _instance
        # Если передан внешний embedder — создаём новый экземпляр (для Ray акторов)
        if embedder is not None:
            obj = super().__new__(cls)
            obj._init(embedder)
            return obj
        # Иначе — синглтон для основного процесса
        if _instance is None:
            logger.info("[MilvusApiClient] инициализация")
            obj = super().__new__(cls)
            connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
            obj._init(EmbeddingService())
            _instance = obj
            logger.info("[MilvusApiClient] готов")
        else:
            logger.debug("[MilvusApiClient] используем существующий экземпляр")
        return _instance

    def _init(self, embedder: EmbeddingService):
        self._collection = Collection(MILVUS_COLLECTION)
        self._collection.load()
        self._embedder = embedder
        self._searcher = MilvusSearcher(
            collection=self._collection,
            embedder=self._embedder,
        )

    def search(
        self,
        text: str,
        mode: str = "max_score",
        level: Optional[List[int]] = None,
        limit: int = 10,
        use_summary: bool = False,
        filter_expr: Optional[str] = None,
        dense_weight: float = 0.5,
        summary_weight: float = 0.3,
        sparse_weight: float = 0.2,
    ) -> Dict[str, Any]:
        if level is None:
            level = [0, 1, 2]

        query = SearchQuery(
            text=text,
            mode=mode,
            level=level,
            limit=limit,
            use_summary=use_summary,
            filter_expr=filter_expr,
            dense_weight=dense_weight,
            summary_weight=summary_weight,
            sparse_weight=sparse_weight,
        )

        if mode == "max_score":
            embeddings = self._embedder([text])
            emb_sparse = embeddings["sparse"]
            logger.info(
                f"[SPARSE] type={type(emb_sparse).__name__}"
                f"  shape={getattr(emb_sparse, 'shape', None)}"
                f"  nnz={getattr(emb_sparse, 'nnz', None)}"
                f"  sample={dict(list(zip(emb_sparse.indices, emb_sparse.data))[:5]) if hasattr(emb_sparse, 'indices') else str(emb_sparse)[:100]}"
            )
            results = self._searcher.search_max_score(
                query=query,
                emb_dense=embeddings["dense"][0],
                emb_sparse=emb_sparse,
            )
        else:
            results = self._searcher.search(query)

        return {"results": results}

    def query(
        self,
        filter_expr: Optional[str] = None,
        limit: int = 10,
        output_fields: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if output_fields is None:
            output_fields = [
                "doc_id", "doc_title", "section_id", "main_section_id",
                "subsection_id", "level", "title", "number", "text",
            ]

        kwargs: Dict[str, Any] = {"limit": limit, "output_fields": output_fields}
        if filter_expr:
            kwargs["expr"] = filter_expr

        rows = self._collection.query(**kwargs)
        return {"results": rows}