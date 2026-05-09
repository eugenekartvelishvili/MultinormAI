import logging
import torch
from typing import List

from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from code.ingestion import config

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self):
        logger.info(
            f"🧠 Загружаем embedding модель: {config.EMBED_MODEL} "
            f"({config.EMBED_DEVICE}, fp16={config.EMBED_USE_FP16})"
        )

        self.ef = BGEM3EmbeddingFunction(
            model_name=config.EMBED_MODEL,
            use_fp16=config.EMBED_USE_FP16,
            device='cpu',
            return_sparse=True,
            return_dense=True,
            return_colbert_vecs=False,
        )

        self.dense_dim = self.ef.dim["dense"]
        self.sparse_dim = self.ef.dim["sparse"]

        logger.info(f"📐 Embedding dims: dense={self.dense_dim}, sparse=dynamic")

    def __call__(self, texts: List[str]) -> dict:
        if not texts:
            return {"dense": [], "sparse": []}
        return self.ef(texts)

    def load(self):
        """Загружаем модель на GPU"""
        if config.EMBED_DEVICE == "cuda":
            self.ef.model.model.to("cuda")
            logger.info("⚡ Embedding модель загружена на GPU")

    def unload(self):
        """Выгружаем модель с GPU"""
        if config.EMBED_DEVICE == "cuda":
            self.ef.model.model.to("cpu")
            torch.cuda.empty_cache()
            logger.info("💤 Embedding модель выгружена с GPU")


def create_embedder() -> EmbeddingService:
    return EmbeddingService()