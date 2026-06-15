"""Backtest the exotic-EV recommendation strategy on real results.

Question: do the +EV 複勝/ワイド/馬連 bets our recommender would make actually
return money? We use closing odds (the model ≈ market, so market-implied win
probabilities 1/odds are a fair, cheap proxy — no per-horse enrich needed),
run the same Harville→EV logic, bet every candidate with prob≥MIN_PROB and
EV≥0 at flat stake, and settle against the actual finishing order.

Conservative: 複勝/ワイド returns use the lower-bound odds (what EV was computed
on); true payouts are a bit higher. Network. Run from repo root:
    python docs/analysis/backtest_exotic_ev.py
"""
import time
from collections import defaultdict
from keiba.race_list import fetch_kaisai_dates, fetch_race_ids
from keiba.result_page import fetch_result_rows
from keiba.odds_page import fetch_odds
from keiba.predictor import (place_k, place_probabilities,
                             quinella_probabilities, wide_probabilities)

MONTHS = [(2026, 5), (2026, 6), (2025, 12)]
MAX_RACES = 250
MIN_PROB = 0.02
STAKE = 100.0


def race_ids():
    dates = []
    for y, m in MONTHS:
        try:
            dates.extend(fetch_kaisai_dates(y, m))
        except Exception as exc:  # noqa: BLE001
            print(f"  ! {y}-{m}: {exc}")
        time.sleep(0.3)
    rids = []
    for d in sorted(set(dates)):
        try:
            rids.extend(fetch_race_ids(d))
        except Exception as exc:  # noqa: BLE001
            print(f"  ! {d}: {exc}")
        time.sleep(0.3)
        if len(set(rids)) >= MAX_RACES:
            break
    return sorted(set(rids))[:MAX_RACES]


def settle_race(odds, rows, spent, ret, hits, count):
    finish = {r["number"]: r["finish"] for r in rows
              if r.get("number") and r.get("finish")}
    if len(finish) < 6 or not odds["win"]:
        return False
    field = len(finish)
    k = place_k(field)
    first = next((n for n, f in finish.items() if f == 1), None)
    second = next((n for n, f in finish.items() if f == 2), None)
    if first is None or second is None:
        return False
    topk = {n for n, f in finish.items() if f <= (k or 2)}
    win_pair = frozenset((first, second))

    z = sum(1.0 / o for o in odds["win"].values())
    wp = {n: (1.0 / o) / z for n, o in odds["win"].items()}
    quin = quinella_probabilities(wp)
    wide = wide_probabilities(wp)
    place = place_probabilities(wp, k) if k else {}

    def bet(bet_type, p, o, won):
        if p is None or p < MIN_PROB or p * o - 1.0 < 0:
            return
        spent[bet_type] += STAKE
        count[bet_type] += 1
        if won:
            ret[bet_type] += o * STAKE
            hits[bet_type] += 1

    for n, o in odds["win"].items():
        bet("単勝", wp.get(n), o, n == first)
    for n, (lo, _hi) in odds["place"].items():
        bet("複勝", place.get(n), lo, n in topk)
    for (a, b), o in odds["quinella"].items():
        bet("馬連", quin.get((a, b)), o, frozenset((a, b)) == win_pair)
    for (a, b), o in odds["wide"].items():
        bet("ワイド", wide.get((a, b)), o, a in topk and b in topk)
    return True


def main():
    spent, ret = defaultdict(float), defaultdict(float)
    hits, count = defaultdict(int), defaultdict(int)
    used = 0
    for rid in race_ids():
        try:
            odds = fetch_odds(rid)
            time.sleep(0.2)
            rows = fetch_result_rows(rid)
            time.sleep(0.2)
        except Exception:  # noqa: BLE001
            continue
        if settle_race(odds, rows, spent, ret, hits, count):
            used += 1

    print(f"\n使用レース数: {used}  (MIN_PROB={MIN_PROB}, 下限オッズ採用=保守的)\n")
    print(f"{'券種':>6} {'点数':>6} {'的中':>5} {'的中率':>7} {'回収率':>8}")
    tot_s = tot_r = 0.0
    for bt in ("単勝", "複勝", "馬連", "ワイド"):
        s, r, h, c = spent[bt], ret[bt], hits[bt], count[bt]
        tot_s += s
        tot_r += r
        if c == 0:
            print(f"{bt:>6} {0:>6} {'-':>5} {'-':>7} {'-':>8}")
            continue
        print(f"{bt:>6} {c:>6} {h:>5} {h / c * 100:>6.1f}% {r / s * 100:>7.1f}%")
    if tot_s:
        print(f"{'合計':>6} {sum(count.values()):>6} {sum(hits.values()):>5} "
              f"{'':>7} {tot_r / tot_s * 100:>7.1f}%")
    print("\n注: 単勝は市場確率だと全馬EVほぼ一定(<0)なので基本ベットされない。")
    print("    +EVが出るのは連系/複系プールが単勝市場と食い違ったとき。")


if __name__ == "__main__":
    main()
