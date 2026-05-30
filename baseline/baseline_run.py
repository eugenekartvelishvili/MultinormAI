"""
baseline_run.py — запускается ВНУТРИ контейнера multinorm
docker exec -it multinorm python /app/baseline/baseline_run.py --questions /app/baseline/questions.json
"""
import json, argparse, time, requests
import numpy as np

# Внутри Docker-сети Ollama доступен по имени сервиса
# ollama-0 и ollama-1 не в одной сети с multinorm — обращаемся через хост
# 172.17.0.1 — стандартный gateway Docker bridge, доступен из любого контейнера
OLLAMA_URL  = "http://172.17.0.1:11434/api/chat"
LLM_MODEL   = "qwen3:4b-instruct"
MODEL_PATH  = "/app/models/bge-m3"   # путь внутри контейнера multinorm

SYSTEM_PROMPT = """Ты — ассистент по нормативно-технической документации предприятия.
Отвечай строго на основе предоставленного контекста.
Если ответ есть в контексте — дай его чётко и укажи источник.
Если ответа нет — скажи что не нашёл информации.
Не придумывай факты которых нет в контексте."""

# ── Модель загружается один раз ─────────────────────────────────────────────
_model = None

def get_model():
    global _model
    if _model is None:
        from FlagEmbedding import BGEM3FlagModel
        print(f"Загружаем BGE-M3 из {MODEL_PATH}...")
        _model = BGEM3FlagModel(MODEL_PATH, use_fp16=True, device="cuda")
        print("Модель готова")
    return _model

def embed(texts: list[str]) -> np.ndarray:
    model = get_model()
    out = model.encode(texts, batch_size=16, max_length=512)
    vecs = np.array(out["dense_vecs"], dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.where(norms == 0, 1, norms)

# ── Чанкование ───────────────────────────────────────────────────────────────
def html_to_text(path: str) -> str:
    import re
    with open(path, encoding="utf-8", errors="ignore") as f:
        raw = f.read()
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script","style","head"]): tag.decompose()
        text = soup.get_text(separator="\n")
    except ImportError:
        text = re.sub(r"<[^>]+>", " ", raw)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return "\n".join(lines)

def make_chunks(docs_dir: str, chunk_size=512, overlap=50) -> list[dict]:
    from pathlib import Path
    chunks = []
    for f in sorted(Path(docs_dir).rglob("*.html")):
        text   = html_to_text(str(f))
        tokens = text.split()
        start  = 0
        idx    = 0
        while start < len(tokens):
            end   = min(start + chunk_size, len(tokens))
            chunk = " ".join(tokens[start:end])
            if len(chunk.strip()) > 30:
                chunks.append({"chunk_id": f"{f.stem}__{idx:04d}",
                               "doc_name": f.stem, "text": chunk})
                idx += 1
            start += chunk_size - overlap
        print(f"  {f.stem}: {len(tokens)} токенов → {idx} чанков")
    return chunks

# ── Поиск ────────────────────────────────────────────────────────────────────
def cosine_search(q_vec: np.ndarray, emb: np.ndarray, k=3):
    scores = emb @ q_vec
    ids    = np.argsort(scores)[::-1][:k]
    return ids.tolist(), scores[ids].tolist()

# ── LLM ──────────────────────────────────────────────────────────────────────
def ask_llm(context: str, question: str) -> tuple[str, float]:
    import re
    payload = {
        "model": LLM_MODEL,
        "stream": False,
        "options": {"num_ctx": 8192},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Контекст:\n{context}\n\nВопрос: {question} /no_think"}
        ]
    }
    t0   = time.time()
    resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
    resp.raise_for_status()
    elapsed = round(time.time() - t0, 2)
    answer = resp.json()["message"]["content"]
    # Убираем <think>...</think> блоки которые Qwen3 иногда добавляет
    answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
    return answer, elapsed

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--questions", default="/app/code/baseline/questions.json")
    p.add_argument("--docs_dir",  default="/app/data/html",
                   help="Папка с HTML-файлами документов")
    p.add_argument("--output",    default="/app/code/baseline/baseline_results.json")
    p.add_argument("--top_k",     type=int, default=3)
    args = p.parse_args()

    with open(args.questions, encoding="utf-8") as f:
        questions = json.load(f)
    print(f"Вопросов: {len(questions)}")

    # Шаг 1: чанки
    print("\n── Шаг 1: чанкование ──")
    chunks = make_chunks(args.docs_dir)
    print(f"Итого чанков: {len(chunks)}")

    # Шаг 2: эмбеддинги
    print("\n── Шаг 2: эмбеддинги ──")
    texts = [c["text"] for c in chunks]
    emb   = embed(texts)
    print(f"Матрица: {emb.shape}")

    # Шаг 3: прогон вопросов
    print("\n── Шаг 3: прогон вопросов ──")
    results = []
    for i, q in enumerate(questions):
        qtext  = q["question"]
        qtype  = q.get("type", "unknown")
        print(f"\n[{i+1}/{len(questions)}] [{qtype}] {qtext[:70]}...")

        q_vec = embed([qtext])[0]
        ids, scores = cosine_search(q_vec, emb, args.top_k)
        top   = [chunks[j] for j in ids]

        context     = "\n\n---\n\n".join(c["text"] for c in top)
        ctx_chars   = len(context)
        answer, llm_t = ask_llm(context, qtext)

        print(f"   ctx={ctx_chars} симв | llm={llm_t}с")
        print(f"   ответ: {answer[:120]}...")

        results.append({
            "id":            q.get("id", i+1),
            "type":          qtype,
            "question":      qtext,
            # --- baseline ---
            "baseline_answer":  answer,
            "baseline_time_s":  llm_t,
            "baseline_ctx_chars": ctx_chars,
            "baseline_chunks":  [{"id": c["chunk_id"], "score": round(float(s),4)}
                                  for c,s in zip(top, scores)],
            # --- иерархия (из твоего лога) ---
            "hier_answer":   q.get("hier_answer", ""),
            "hier_time_s":   q.get("hier_time_s", None),
            # --- оценки (заполнить руками после прогона) ---
            "score_baseline": None,   # 0=плохо  1=частично  2=хорошо
            "score_hier":     None,
            "notes":          ""
        })

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Сводка по времени
    times = [r["baseline_time_s"] for r in results]
    ctxs  = [r["baseline_ctx_chars"] for r in results]
    print(f"\n{'='*50}")
    print(f"Baseline готов ({len(results)} вопросов)")
    print(f"  Среднее время LLM:    {sum(times)/len(times):.2f}с")
    print(f"  Медиана:              {sorted(times)[len(times)//2]:.2f}с")
    print(f"  Средний контекст:     {sum(ctxs)//len(ctxs)} симв")
    print(f"\nТеперь:")
    print(f"  1. Открой {args.output}")
    print(f"  2. Для каждого вопроса заполни score_baseline и score_hier (0/1/2)")
    print(f"  3. Запусти analyze_results.py")

if __name__ == "__main__":
    main()