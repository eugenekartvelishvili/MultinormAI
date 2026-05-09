"""
baseline_chunk.py — шаг 1: читаем HTML-файлы, режем на чанки 512 токенов
Использование: python baseline_chunk.py --docs_dir /path/to/html --output chunks.json
"""
import os, re, json, argparse
from pathlib import Path

try:
    from bs4 import BeautifulSoup
    BS_OK = True
except ImportError:
    BS_OK = False

def html_to_text(html_path: str) -> str:
    with open(html_path, encoding="utf-8", errors="ignore") as f:
        raw = f.read()
    if BS_OK:
        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script", "style", "head"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
    else:
        # Fallback: просто убираем теги регуляркой
        text = re.sub(r"<[^>]+>", " ", raw)
    # Нормализуем пробелы
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return "\n".join(lines)

def simple_tokenize(text: str) -> list[str]:
    """Простая токенизация по пробелам — ~1 токен ≈ 0.75 слова,
    для русского берём слова как токены (грубо но достаточно)."""
    return text.split()

def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> list[str]:
    tokens = simple_tokenize(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk = " ".join(tokens[start:end])
        if len(chunk.strip()) > 20:  # Не добавляем пустые чанки
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs_dir", required=True, help="Папка с HTML-файлами")
    parser.add_argument("--output", default="chunks.json", help="Выходной файл")
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=50)
    args = parser.parse_args()

    docs_dir = Path(args.docs_dir)
    html_files = list(docs_dir.rglob("*.html")) + list(docs_dir.rglob("*.htm"))
    
    if not html_files:
        print(f"HTML-файлы не найдены в {docs_dir}")
        return

    all_chunks = []
    for html_file in sorted(html_files):
        text = html_to_text(str(html_file))
        if not text.strip():
            continue
        chunks = chunk_text(text, args.chunk_size, args.overlap)
        doc_name = html_file.stem
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "chunk_id": f"{doc_name}__chunk_{i:04d}",
                "doc_name": doc_name,
                "chunk_index": i,
                "text": chunk,
                "char_len": len(chunk)
            })
        print(f"  {doc_name}: {len(text)} симв → {len(chunks)} чанков")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    
    print(f"\nИтого: {len(html_files)} документов → {len(all_chunks)} чанков")
    print(f"Сохранено в: {args.output}")

if __name__ == "__main__":
    main()
"""
baseline_chunk.py — шаг 1: читаем HTML-файлы, режем на чанки 512 токенов
Использование: python baseline_chunk.py --docs_dir /path/to/html --output chunks.json
"""
import os, re, json, argparse
from pathlib import Path

try:
    from bs4 import BeautifulSoup
    BS_OK = True
except ImportError:
    BS_OK = False

def html_to_text(html_path: str) -> str:
    with open(html_path, encoding="utf-8", errors="ignore") as f:
        raw = f.read()
    if BS_OK:
        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script", "style", "head"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
    else:
        # Fallback: просто убираем теги регуляркой
        text = re.sub(r"<[^>]+>", " ", raw)
    # Нормализуем пробелы
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return "\n".join(lines)

def simple_tokenize(text: str) -> list[str]:
    """Простая токенизация по пробелам — ~1 токен ≈ 0.75 слова,
    для русского берём слова как токены (грубо но достаточно)."""
    return text.split()

def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> list[str]:
    tokens = simple_tokenize(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk = " ".join(tokens[start:end])
        if len(chunk.strip()) > 20:  # Не добавляем пустые чанки
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs_dir", required=True, help="Папка с HTML-файлами")
    parser.add_argument("--output", default="chunks.json", help="Выходной файл")
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=50)
    args = parser.parse_args()

    docs_dir = Path(args.docs_dir)
    html_files = list(docs_dir.rglob("*.html")) + list(docs_dir.rglob("*.htm"))
    
    if not html_files:
        print(f"HTML-файлы не найдены в {docs_dir}")
        return

    all_chunks = []
    for html_file in sorted(html_files):
        text = html_to_text(str(html_file))
        if not text.strip():
            continue
        chunks = chunk_text(text, args.chunk_size, args.overlap)
        doc_name = html_file.stem
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "chunk_id": f"{doc_name}__chunk_{i:04d}",
                "doc_name": doc_name,
                "chunk_index": i,
                "text": chunk,
                "char_len": len(chunk)
            })
        print(f"  {doc_name}: {len(text)} симв → {len(chunks)} чанков")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    
    print(f"\nИтого: {len(html_files)} документов → {len(all_chunks)} чанков")
    print(f"Сохранено в: {args.output}")

if __name__ == "__main__":
    main()