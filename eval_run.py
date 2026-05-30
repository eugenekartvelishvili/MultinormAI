#!/usr/bin/env python3
"""
eval_run.py — прогон вопросов через RAG-систему и запись результатов.

Запуск:
    docker cp eval_run.py multinorm:/app/eval_run.py
    docker exec -it multinorm python3 /app/eval_run.py --api http://localhost:8000
    docker exec -it multinorm python3 /app/eval_run.py --limit 5
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

QUESTIONS = [
    {"id": 1,  "question": "Что такое риск?", "question_type": "definition"},
    {"id": 2,  "question": "Какие задачи у отдела геологии?", "question_type": "broad_overview"},
    {"id": 3,  "question": "Какие задачи у отдела бурения и проектирования строительства скважин?", "question_type": "broad_overview"},
    {"id": 4,  "question": "В чем разница между накоплением и хранением отходов?", "question_type": "definition"},
    {"id": 5,  "question": "Какие документы нужно получить от поставщика после поставки?", "question_type": "fact"},
    {"id": 6,  "question": "Нужно ли ежегодно проводить переоценку поставщиков?", "question_type": "fact"},
    {"id": 7,  "question": "Есть ли у нас регламент по обращению с отходами?", "question_type": "norm_reference"},
    {"id": 8,  "question": "Можно ли закупить оборудование без процедуры оценки поставщика?", "question_type": "fact"},
    {"id": 9,  "question": "Есть ли ограничения по сроку хранения отходов I класса?", "question_type": "fact"},
    {"id": 10, "question": "Как подготовить тендерное предложение?", "question_type": "procedure"},
    {"id": 11, "question": "Кто такой федеральный оператор в контексте отходов?", "question_type": "definition"},
    {"id": 12, "question": "Есть ли инструкция по внешним поставкам?", "question_type": "norm_reference"},
    {"id": 13, "question": "Какие есть классы опасности отходов?", "question_type": "fact"},
    {"id": 14, "question": "Сколько можно хранить отходы у себя на территории?", "question_type": "fact"},
    {"id": 15, "question": "Что делать если поставщик привез продукцию ненадлежащего качества?", "question_type": "procedure"},
    {"id": 16, "question": "Что такое идентификация экологических аспектов?", "question_type": "definition"},
    {"id": 17, "question": "Кто отвечает за учёт отходов?", "question_type": "fact"},
    {"id": 18, "question": "Есть ли у нас положение о коммерческой тайне?", "question_type": "norm_reference"},
    {"id": 19, "question": "Какой минимальный объем оперативной памяти должен быть у покупаемых компьютеров?", "question_type": "fact"},
    {"id": 20, "question": "Какой порядок действий при внешних поставках?", "question_type": "procedure"},
    {"id": 21, "question": "Что такое коммерческая тайна?", "question_type": "definition"},
    {"id": 22, "question": "Каик сведения являются коммерческой тайной?", "question_type": "fact"},
    {"id": 23, "question": "куда нужно класть файлы при регистрации", "question_type": "fact"},
    {"id": 24, "question": "Кому подчиняется полевая партия?", "question_type": "fact"},
    {"id": 25, "question": "какой билет на самолет можно брать в командировки", "question_type": "fact"},
    {"id": 26, "question": "Какие задачи у финансового отдела?", "question_type": "broad_overview"},
    {"id": 27, "question": "Какая структура каталогов диска P?", "question_type": "fact"},
    {"id": 28, "question": "Что входит в персональные данные сотрудников?", "question_type": "definition"},
    {"id": 29, "question": "Что будет если превысить лимиты корпоративной мобильной связи?", "question_type": "fact"},
    {"id": 30, "question": "Должен ли отдел инженерных изысканий создавать проектную документацию?", "question_type": "fact"},
    {"id": 31, "question": "Какие у нас стадии выполнения работ?", "question_type": "broad_overview"},
    {"id": 32, "question": "Как оценить влияние экологических аспектов?", "question_type": "procedure"},
    {"id": 33, "question": "Что нужно делать на предпроектной проработке?", "question_type": "procedure"},
    {"id": 34, "question": "Какого класса билет на самолет можно купить при командировке?", "question_type": "fact"},
    {"id": 35, "question": "Что включается в бюджет командировки?", "question_type": "fact"},
    {"id": 36, "question": "Какие правила для транспортных расходов в командировке?", "question_type": "fact"},
    {"id": 37, "question": "Есть ли положение о полевой партии?", "question_type": "norm_reference"},
    {"id": 38, "question": "Есть ли ограничения по сроку хранения отходов?", "question_type": "fact"},
    {"id": 39, "question": "Есть ли у нас рабочая инструкция по структуре каталогов?", "question_type": "norm_reference"},
    {"id": 40, "question": "В каком документе описаны правила командировок?", "question_type": "norm_reference"},
    {"id": 41, "question": "Есть ли у нас инструкция по регистрации и архивированию?", "question_type": "norm_reference"},
    {"id": 42, "question": "Есть ли регламент по защите персональных данных?", "question_type": "norm_reference"},
    {"id": 43, "question": "Есть ли у нас положение о корпоративной мобильной связи?", "question_type": "norm_reference"},
    {"id": 44, "question": "Кто несет ответственность за соблюдение правил обращения с персональными данными?", "question_type": "fact"},
    {"id": 45, "question": "Есть ли у нас инструкция по управлению рисками?", "question_type": "norm_reference"},
    {"id": 46, "question": "Есть ли регламент по командировкам?", "question_type": "norm_reference"},
    {"id": 47, "question": "Как проходит оценка поставщика при первой закупке?", "question_type": "procedure"},
    {"id": 48, "question": "Какие у нас правила служебного поведения?", "question_type": "broad_overview"},
    {"id": 49, "question": "Что такое верификация внешней поставки?", "question_type": "definition"},
    {"id": 50, "question": "Что такое субподрядная работа при поставках?", "question_type": "definition"},
    {"id": 51, "question": "Что такое процесс управления рисками?", "question_type": "broad_overview"},
    {"id": 52, "question": "Как оформить заявку на командировку", "question_type": "procedure"},
    {"id": 53, "question": "Какие задачи у полевой партии", "question_type": "broad_overview"},
    {"id": 54, "question": "Как транспортировать отходы IV класса опасности?", "question_type": "procedure"},
    {"id": 55, "question": "Что значит накопление отходов?", "question_type": "definition"},
    {"id": 56, "question": "Что делать сотруднику по окончании командировки?", "question_type": "procedure"},
    {"id": 57, "question": "Какие критерии используются при первичной оценке поставщика?", "question_type": "fact"},
    {"id": 58, "question": "Какие этапы включает стадия проектирования?", "question_type": "broad_overview"},
    {"id": 59, "question": "Есть ли у нас кодекс этики?", "question_type": "norm_reference"},
    {"id": 60, "question": "Где описаны персональные данные?", "question_type": "norm_reference"},
    {"id": 61, "question": "В каком документе описаны закупки?", "question_type": "norm_reference"},
    {"id": 62, "question": "Что такое одобренный поставщик?", "question_type": "definition"},
    {"id": 63, "question": "Кто утверждает структуру полевой партии?", "question_type": "fact"},
    {"id": 64, "question": "Нужна ли лицензия для транспортировки отходов III класса?", "question_type": "fact"},
    {"id": 65, "question": "Какой порядок действий при внешних поставках", "question_type": "procedure"},
    {"id": 66, "question": "Надо ли каждый год переоценивать поставщиков?", "question_type": "fact"},
    {"id": 67, "question": "Какие задачи у отдела бурения?", "question_type": "broad_overview"},
    {"id": 68, "question": "Есть ли у нас положение о командировках?", "question_type": "norm_reference"},
]


def ask_via_api(question: str, api_url: str, timeout: int = 120) -> dict:
    import urllib.request, urllib.error
    payload = json.dumps({"text": question}).encode("utf-8")
    req = urllib.request.Request(
        f"{api_url.rstrip('/')}/ask",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.reason}"}
    except Exception as e:
        return {"error": str(e)}


def ask_via_router(question: str) -> dict:
    sys.path.insert(0, "/app")
    try:
        from code.retrieval.rag.router import adaptive_search
        t0 = time.time()
        result = adaptive_search(question)
        result["elapsed"] = time.time() - t0
        return result
    except Exception as e:
        return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Прогон вопросов через RAG-систему")
    parser.add_argument("--api", default=None,
                        help="URL API (http://localhost:8000). Если не указан — прямой вызов router.")
    parser.add_argument("--questions", default=None,
                        help="JSON файл с вопросами")
    parser.add_argument("--output", default=None,
                        help="Куда писать результаты")
    parser.add_argument("--delay", type=float, default=4.0,
                        help="Пауза между запросами в секундах (default: 2)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Ограничить количество вопросов (для теста)")
    args = parser.parse_args()

    if args.questions and Path(args.questions).exists():
        with open(args.questions, encoding="utf-8") as f:
            questions = json.load(f)
        print(f"Загружено {len(questions)} вопросов из {args.questions}")
    else:
        questions = QUESTIONS
        print(f"Встроенный список: {len(questions)} вопросов")

    if args.limit:
        questions = questions[:args.limit]
        print(f"Ограничено до {len(questions)} вопросов")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output or f"/app/data/eval_results_{ts}.json"

    print(f"\nРежим: {'API ' + args.api if args.api else 'прямой вызов router'}")
    print(f"Результаты → {output_path}")
    print(f"Всего: {len(questions)} вопросов\n")
    print("=" * 70)

    results = []
    errors  = 0

    for i, q in enumerate(questions):
        qid      = q.get("id", i + 1)
        question = q["question"]
        qtype    = q.get("question_type", "unknown")

        print(f"[{i+1:2d}/{len(questions)}] id={qid} [{qtype}] {question[:65]}...")

        t0  = time.time()
        raw = ask_via_api(question, args.api) if args.api else ask_via_router(question)
        elapsed = time.time() - t0

        if "error" in raw:
            answer      = f"ОШИБКА: {raw['error']}"
            sources     = []
            context_len = 0
            errors     += 1
            status      = "ERROR"
        else:
            answer      = raw.get("answer", "")
            sources     = raw.get("sources", [])
            context_len = len(raw.get("context", ""))
            status      = "OK"

        result = {
            "id":            qid,
            "question":      question,
            "question_type": qtype,
            "answer":        answer,
            "sources":       sources,
            "context_len":   context_len,
            "elapsed_s":     round(elapsed, 2),
            "status":        status,
            "detected_type": raw.get("question_type", ""),
        }
        results.append(result)

        print(f"         {'✓' if status=='OK' else '✗'} {elapsed:.1f}s  "
              f"ctx={context_len}  src={len(sources)}  {status}")
        if answer and status == "OK":
            print(f"         → {answer.replace(chr(10), ' ')[:120]}...")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        if i < len(questions) - 1:
            time.sleep(args.delay)

    print("\n" + "=" * 70)
    ok = [r for r in results if r["status"] == "OK"]
    print(f"ИТОГО: {len(results)} вопросов, ошибок: {errors}")
    if ok:
        avg_t   = sum(r["elapsed_s"] for r in ok) / len(ok)
        avg_ctx = sum(r["context_len"] for r in ok) / len(ok)
        total_t = sum(r["elapsed_s"] for r in results)
        print(f"Среднее время: {avg_t:.1f}s  |  Средний контекст: {avg_ctx:.0f} симв.")
        print(f"Общее время: {total_t:.0f}s ({total_t/60:.1f} мин)")

        by_type = {}
        for r in ok:
            by_type.setdefault(r["question_type"], []).append(r["elapsed_s"])
        print("\nВремя по типам:")
        for t in ["fact", "definition", "procedure", "broad_overview", "norm_reference"]:
            v = by_type.get(t, [])
            if v:
                print(f"  {t:<20} n={len(v):2d}  avg={sum(v)/len(v):.1f}s  "
                      f"min={min(v):.1f}s  max={max(v):.1f}s")

    txt_path = output_path.replace(".json", ".txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Eval run: {ts}\n")
        f.write(f"Вопросов: {len(results)}, ошибок: {errors}\n\n")
        for r in results:
            f.write(f"{'='*70}\n")
            f.write(f"[{r['id']:2d}] [{r['question_type']}] {r['question']}\n")
            f.write(f"Время: {r['elapsed_s']}s  Контекст: {r['context_len']} симв.\n")
            if r["sources"]:
                f.write(f"Источники: {', '.join(str(s) for s in r['sources'][:3])}\n")
            f.write(f"\nОТВЕТ:\n{r['answer']}\n\n")

    print(f"\nJSON:  {output_path}")
    print(f"Текст: {txt_path}")


if __name__ == "__main__":
    main()