# main.py
import os
from pathlib import Path
from pipeline.runner import run_pipeline

if __name__ == "__main__":
    data_dir = Path("data").resolve()
    pdf_files = [(data_dir / f).resolve() for f in os.listdir(data_dir) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print("❌ Нет PDF файлов в папке data/")
    else:
        print(f"📄 Найдено {len(pdf_files)} PDF файлов. Запускаем пайплайн...")
        run_pipeline(pdf_files)