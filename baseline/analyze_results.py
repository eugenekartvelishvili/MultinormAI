"""
analyze_results.py — считаем метрики H2 и H3 после прогона baseline
Использование: python analyze_results.py --results baseline_results.json

Как заполнить результаты перед запуском:
  Открой baseline_results.json, для каждого вопроса заполни:
    "score_baseline": 0/1/2  (0=плохо, 1=частично, 2=хорошо)
    "score_hier":     0/1/2
    "hier_answer":    ответ твоей системы (скопировать из лога/Bitrix)
    "hier_time_s":    время ответа твоей системы в секундах
  
  Критерии оценки (сравнивай с gold_answer и expected_key_points):
    2 = ответ корректный, покрывает ключевые факты, есть ссылка на источник
    1 = ответ частично верный, или верный но без источника, или упущены важные детали
    0 = ответ неверный, галлюцинация, или "не нашёл информации" когда она есть
"""
import json, argparse
from collections import defaultdict

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="/app/baseline/baseline_results.json")
    args = p.parse_args()

    with open(args.results, encoding="utf-8") as f:
        results = json.load(f)

    total = len(results)
    scored = [r for r in results
              if r.get("score_baseline") is not None
              and r.get("score_hier") is not None]

    if not scored:
        print(f"Записей всего: {total}")
        print("Нет оценённых записей.")
        print("Заполни score_baseline и score_hier (0/1/2) в JSON и запусти снова.")
        return

    print(f"\n{'='*60}")
    print(f"РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТА (оценено {len(scored)}/{total})")
    print(f"{'='*60}")

    # ── H2: Время ──────────────────────────────────────────────
    times_b = [r["baseline_time_s"] for r in scored if r.get("baseline_time_s")]
    times_h = [r["hier_time_s"]     for r in scored if r.get("hier_time_s")]

    print(f"\nH2 — Время ответа:")
    if times_b:
        avg_tb = sum(times_b) / len(times_b)
        med_tb = sorted(times_b)[len(times_b)//2]
        print(f"  Baseline  avg={avg_tb:.2f}с  медиана={med_tb:.2f}с")
    if times_h:
        avg_th = sum(times_h) / len(times_h)
        med_th = sorted(times_h)[len(times_h)//2]
        print(f"  Иерархия  avg={avg_th:.2f}с  медиана={med_th:.2f}с")
    if times_b and times_h:
        diff = avg_tb - avg_th
        print(f"  Разница:  {diff:+.2f}с ({'иерархия быстрее' if diff > 0 else 'baseline быстрее'})")
        if avg_th <= 15:
            print(f"  Вывод H2: ПОДТВЕРЖДЕНА — иерархия укладывается в ≤15 сек")
            if diff > 0:
                print(f"            и быстрее baseline на {diff:.1f}с")
        else:
            print(f"  Вывод H2: НЕ подтверждена — иерархия >15 сек")

    # ── H3: Качество ───────────────────────────────────────────
    sb = [r["score_baseline"] for r in scored]
    sh = [r["score_hier"]     for r in scored]
    avg_b  = sum(sb) / len(sb)
    avg_h  = sum(sh) / len(sh)
    pct_b  = avg_b / 2 * 100
    pct_h  = avg_h / 2 * 100
    diff_q = pct_h - pct_b

    print(f"\nH3 — Качество ответов (шкала 0-2):")
    print(f"  Baseline  avg={avg_b:.2f}  ({pct_b:.1f}%)")
    print(f"  Иерархия  avg={avg_h:.2f}  ({pct_h:.1f}%)")
    print(f"  Разница:  {diff_q:+.1f} пп в пользу {'иерархии' if diff_q>0 else 'baseline'}")
    if diff_q >= 15:
        print(f"  Вывод H3: ПОДТВЕРЖДЕНА — разница ≥15 пп")
    elif diff_q > 0:
        print(f"  Вывод H3: ЧАСТИЧНО — иерархия лучше, но разница <15 пп")
    else:
        print(f"  Вывод H3: НЕ подтверждена")

    # ── Контекст ───────────────────────────────────────────────
    ctxs = [r["baseline_ctx_chars"] for r in scored if r.get("baseline_ctx_chars")]
    if ctxs:
        print(f"\nРазмер контекста baseline:")
        print(f"  avg={sum(ctxs)//len(ctxs)} симв  max={max(ctxs)}  min={min(ctxs)}")

    # ── По типам ───────────────────────────────────────────────
    by_type = defaultdict(lambda: {"sb":[],"sh":[],"tb":[],"th":[]})
    for r in scored:
        t = r["type"]
        by_type[t]["sb"].append(r["score_baseline"])
        by_type[t]["sh"].append(r["score_hier"])
        if r.get("baseline_time_s"): by_type[t]["tb"].append(r["baseline_time_s"])
        if r.get("hier_time_s"):     by_type[t]["th"].append(r["hier_time_s"])

    print(f"\nПо типам запросов (качество):")
    print(f"  {'Тип':<18} {'n':>3}  {'Baseline':>9}  {'Иерархия':>9}  {'Разница':>8}")
    print(f"  {'-'*55}")
    for t in ["fact","definition","procedure","norm_reference","broad_overview"]:
        v = by_type.get(t)
        if not v or not v["sb"]: continue
        n  = len(v["sb"])
        pb = sum(v["sb"])/n/2*100
        ph = sum(v["sh"])/n/2*100
        print(f"  {t:<18} {n:>3}  {pb:>8.1f}%  {ph:>8.1f}%  {ph-pb:>+7.1f}пп")

    # ── Победители ─────────────────────────────────────────────
    hier_wins = sum(1 for r in scored if r["score_hier"] > r["score_baseline"])
    base_wins = sum(1 for r in scored if r["score_baseline"] > r["score_hier"])
    ties      = sum(1 for r in scored if r["score_hier"] == r["score_baseline"])
    print(f"\nПобедитель по вопросам:")
    print(f"  Иерархия лучше:  {hier_wins}/{len(scored)} ({hier_wins/len(scored)*100:.0f}%)")
    print(f"  Baseline лучше:  {base_wins}/{len(scored)} ({base_wins/len(scored)*100:.0f}%)")
    print(f"  Ничья:           {ties}/{len(scored)}")

    # ── Детальная таблица ──────────────────────────────────────
    print(f"\nДетальная таблица:")
    print(f"  {'id':>3}  {'тип':<15}  {'doc':<15}  {'Б':>2}  {'И':>2}  {'вопрос'}")
    print(f"  {'-'*75}")
    for r in scored:
        doc = r.get("expected_doc","")[:14]
        q   = r["question"][:40]
        sb  = r["score_baseline"]
        sh  = r["score_hier"]
        win = "←И" if sh>sb else ("←Б" if sb>sh else "  =")
        print(f"  {r['id']:>3}  {r['type']:<15}  {doc:<15}  {sb:>2}  {sh:>2}  {win}  {q}...")

    print(f"\nГотово. Используй эти цифры для раздела диплома «Результаты проверки гипотез».")

if __name__ == "__main__":
    main()