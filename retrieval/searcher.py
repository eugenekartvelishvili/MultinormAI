import logging
import math
from typing import List, Optional, Dict

from pymilvus import Collection
from code.ingestion.embedding import EmbeddingService
from code.retrieval.query import SearchQuery

logger = logging.getLogger(__name__)


class MilvusSearcher:
    def __init__(self, collection: Collection, embedder: EmbeddingService):
        self.collection = collection
        self.embedder = embedder

    # =========================================================
    # Public API
    # =========================================================

    def search(self, query: SearchQuery) -> List[dict]:
        """
        Роутер по режимам поиска.
        mode:
          "dense"      — только основной dense вектор (cosine)
          "sparse"     — только sparse вектор (IP)
          "summary"    — только summary dense вектор (cosine)
          "hybrid"     — взвешенная сумма dense+sparse+summary
          "max_score"  — max(dense, summary, sparse) без весов
        """
        logger.info(
            f"🔍 Search: '{query.text[:80]}'"
            f" mode={query.mode} level={query.levels} limit={query.limit}"
        )

        embeddings = self.embedder([query.text])
        expr = self._build_expr(query.levels, query.filter_expr)

        if query.mode == "dense":
            return self._dense(embeddings["dense"][0], query.limit, expr)

        if query.mode == "sparse":
            return self._sparse(embeddings["sparse"], query.limit, expr)

        if query.mode == "summary":
            return self._summary(embeddings["dense"][0], query.limit, expr)

        if query.mode == "hybrid":
            return self._hybrid(
                dense_emb=embeddings["dense"][0],
                sparse_emb=embeddings["sparse"],
                limit=query.limit,
                expr=expr,
                dense_weight=query.dense_weight,
                sparse_weight=query.sparse_weight,
                summary_weight=query.summary_weight,
                use_summary=query.use_summary,
            )

        if query.mode == "max_score":
            return self.search_max_score(query, embeddings["dense"][0], embeddings["sparse"])

        raise ValueError(f"Unknown mode: {query.mode!r}")

    # =========================================================
    # Одиночные режимы
    # =========================================================

    def _dense(self, emb, limit: int, expr: Optional[str]) -> List[dict]:
        """Поиск только по основному dense вектору."""
        hits = self._search_field(emb, "dense_vector", limit, expr, "COSINE")
        return self._format_and_normalize(hits, "dense")

    def _sparse(self, emb, limit: int, expr: Optional[str]) -> List[dict]:
        """Поиск только по sparse вектору. Score нормализован через sigmoid."""
        hits = self._search_field(emb, "sparse_vector", limit, expr, "IP")
        return self._format_and_normalize(hits, "sparse")

    def _summary(self, emb, limit: int, expr: Optional[str]) -> List[dict]:
        """Поиск только по summary dense вектору."""
        hits = self._search_field(emb, "dense_vector_summary", limit, expr, "COSINE")
        return self._format_and_normalize(hits, "summary")

    # =========================================================
    # Гибридный режим (взвешенная сумма)
    # =========================================================

    def _hybrid(
        self,
        dense_emb,
        sparse_emb,
        limit: int,
        expr: Optional[str],
        dense_weight: float,
        sparse_weight: float,
        summary_weight: float,
        use_summary: bool,
    ) -> List[dict]:
        """
        Взвешенная сумма:
          score = dense_weight*dense + sparse_weight*sparse + summary_weight*summary
        Все компоненты нормализованы в [0,1] перед взвешиванием.
        """
        top_n = limit * 5

        dense_hits = self._search_field(dense_emb, "dense_vector", top_n, expr, "COSINE")
        sparse_hits = self._search_field(sparse_emb, "sparse_vector", top_n, expr, "IP")
        summary_hits = (
            self._search_field(dense_emb, "dense_vector_summary", top_n, expr, "COSINE")
            if use_summary else []
        )

        combined: Dict[str, dict] = {}

        def update(hits, kind):
            for h in hits:
                key = h.get("section_id") or h.get("doc_id")
                norm = self._normalize(h["score"], kind)
                if key not in combined:
                    combined[key] = {**h, "_dense": 0.0, "_sparse": 0.0, "_summary": 0.0}
                combined[key][f"_{kind}"] = max(combined[key][f"_{kind}"], norm)

        update(dense_hits, "dense")
        update(sparse_hits, "sparse")
        if use_summary:
            update(summary_hits, "summary")

        results = []
        for v in combined.values():
            score = (
                dense_weight   * v["_dense"] +
                sparse_weight  * v["_sparse"] +
                summary_weight * v["_summary"]
            )
            v["score"] = score
            logger.info(
                f"[HYBRID] {score:.4f}"
                f" (d={v['_dense']:.2f} sp={v['_sparse']:.2f} sum={v['_summary']:.2f})"
                f" {v.get('title', '')!r}"
            )
            results.append(v)

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    # =========================================================
    # Max score (лучший из трёх каналов)
    # =========================================================

    def search_max_score(self, query: SearchQuery, emb_dense, emb_sparse) -> List[dict]:
        """
        score = max(norm_dense, norm_summary, norm_sparse).
        Каждый канал нормализован независимо через _normalize.
        Основной метод поиска в стратегиях A и B pass2.
        """
        expr = self._build_expr(query.levels, query.filter_expr)
        top_n = query.limit * 5

        dense_hits   = self._search_field(emb_dense,  "dense_vector",         top_n, expr, "COSINE")
        summary_hits = self._search_field(emb_dense,  "dense_vector_summary", top_n, expr, "COSINE")
        sparse_hits  = self._search_field(emb_sparse, "sparse_vector",        top_n, expr, "IP")

        combined: Dict[str, dict] = {}

        for h in dense_hits:
            key = h.get("section_id") or h.get("doc_id")
            norm = self._normalize(h["score"], "dense")
            if key not in combined:
                combined[key] = {**h, "_dense": 0.0, "_summary": 0.0, "_sparse": 0.0}
            combined[key]["_dense"] = max(combined[key]["_dense"], norm)

        for h in summary_hits:
            key = h.get("section_id") or h.get("doc_id")
            norm = self._normalize(h["score"], "summary")
            if key not in combined:
                combined[key] = {**h, "_dense": 0.0, "_summary": 0.0, "_sparse": 0.0}
            combined[key]["_summary"] = max(combined[key]["_summary"], norm)

        for h in sparse_hits:
            key = h.get("section_id") or h.get("doc_id")
            norm = self._normalize(h["score"], "sparse")
            if key not in combined:
                combined[key] = {**h, "_dense": 0.0, "_summary": 0.0, "_sparse": 0.0}
            combined[key]["_sparse"] = max(combined[key]["_sparse"], norm)

        results = []
        for v in combined.values():
            v["score"] = max(v["_dense"], v["_summary"], v["_sparse"])
            logger.info(
                f"[MAX_SCORE] {v['score']:.4f}"
                f" (d={v['_dense']:.2f} sum={v['_summary']:.2f} sp={v['_sparse']:.2f})"
                f" {v.get('title', '')!r}"
            )
            results.append(v)

        results.sort(key=lambda x: x["score"], reverse=True)
        logger.info(f"[MAX_SCORE] итого {len(results)} → топ {query.limit}")
        return results[:query.limit]

    # =========================================================
    # Helpers
    # =========================================================

    def _search_field(
        self,
        emb,
        field: str,
        limit: int,
        expr: Optional[str],
        metric: str,
    ) -> List[dict]:
        hits = self.collection.search(
            [emb],
            field,
            {"metric_type": metric, "params": {}},
            limit=limit,
            expr=expr,
            output_fields=self._output_fields(),
        )[0]
        return self._format_hits(hits)

    def _format_and_normalize(self, hits: List[dict], kind: str) -> List[dict]:
        """Нормализует score для одиночных режимов (dense/sparse/summary)."""
        for h in hits:
            h["score"] = self._normalize(h["score"], kind)
        return hits

    @staticmethod
    def _normalize(score: float, kind: str) -> float:
        """
        Нормализация score в [0, 1]:

        dense / summary — cosine similarity в [-1, 1]:
          norm = (score + 1) / 2

        sparse — inner product в [0, ∞]:
          norm = sigmoid(score) = 1 / (1 + exp(-score))
          При BGE-M3 sparse scores 0.1–5.0 даёт диапазон [0.52, 0.99].
          Хорошее разделение без знания глобального max.
        """
        if kind in ("dense", "summary"):
            return (score + 1.0) / 2.0
        if kind == "sparse":
            return 1.0 / (1.0 + math.exp(-score))
        return score

    @staticmethod
    def _format_hits(hits) -> List[dict]:
        result = []
        for h in hits:
            e = h.entity
            result.append({
                "score":            float(h.score),
                "text":             e.get("text"),
                "level":            e.get("level"),
                "doc_id":           e.get("doc_id"),
                "doc_title":        e.get("doc_title"),
                "section_id":       e.get("section_id"),
                "main_section_id":  e.get("main_section_id"),
                "subsection_id":    e.get("subsection_id"),
                "title":            e.get("title"),
                "number":           e.get("number"),
            })
        return result

    @staticmethod
    def _build_expr(levels, filter_expr: Optional[str]) -> Optional[str]:
        level_part = None
        if levels:
            level_part = "level in [" + ",".join(map(str, levels)) + "]"
        if level_part and filter_expr:
            return f"({level_part}) && ({filter_expr})"
        return level_part or filter_expr or None

    @staticmethod
    def _output_fields() -> List[str]:
        return [
            "text", "level", "doc_id", "doc_title",
            "section_id", "main_section_id", "subsection_id",
            "title", "number",
        ]