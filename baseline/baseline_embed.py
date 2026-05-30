"""
baseline_embed.py — шаг 2: генерируем dense-эмбеддинги для чанков через BGE-M3
Использование: python baseline_embed.py --chunks chunks.json --model_path /app/models/bge-m3 --output embeddings.npy
"""
import json, argparse, time
import numpy as np
from pathlib import Path

def load_model(model_path: str):
    """Загружаем BGE-M3 — та же модель что в твоей системе."""
    try:
        from FlagEmbedding import BGEM3FlagModel
        print(f"Загружаем BGE-M3 из {model_path}...")
        model = BGEM3FlagModel(model_path, use_fp16=True, device="cuda")
        print("Модель загружена (CUDA)")
        return model, "flagembedding"
    except ImportError:
        pass
    
    try:
        from sentence_transformers import SentenceTransformer
        print(f"Загружаем через sentence-transformers из {model_path}...")
        model = SentenceTransformer(model_path, device="cuda")
        print("Модель загружена (sentence-transformers)")
        return model, "sbert"
    except ImportError:
        raise RuntimeError("Нужен FlagEmbedding или sentence-transformers")

def embed_texts(model, texts: list[str], backend: str, batch_size: int = 32) -> np.ndarray:
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        if backend == "flagembedding":
            out = model.encode(batch, batch_size=batch_size, max_length=512)
            vecs = out["dense_vecs"]
        else:
            vecs = model.encode(batch, batch_size=batch_size, normalize_embeddings=True)
        all_vecs.append(np.array(vecs, dtype=np.float32))
        print(f"  Batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1} готов ({len(batch)} текстов)")
    return np.vstack(all_vecs)

def cosine_normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return vecs / norms

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", default="chunks.json")
    parser.add_argument("--model_path", default="/app/models/bge-m3")
    parser.add_argument("--output", default="embeddings.npy")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    with open(args.chunks, encoding="utf-8") as f:
        chunks = json.load(f)
    
    texts = [c["text"] for c in chunks]
    print(f"Чанков для эмбеддинга: {len(texts)}")

    model, backend = load_model(args.model_path)
    
    t0 = time.time()
    embeddings = embed_texts(model, texts, backend, args.batch_size)
    embeddings = cosine_normalize(embeddings)
    elapsed = time.time() - t0

    np.save(args.output, embeddings)
    print(f"\nЭмбеддинги: shape={embeddings.shape}, dtype={embeddings.dtype}")
    print(f"Время: {elapsed:.1f}с")
    print(f"Сохранено в: {args.output}")

if __name__ == "__main__":
    main()
