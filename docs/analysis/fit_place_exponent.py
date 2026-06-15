"""Fit a Harville place-exponent λ against real finishing order.

Harville assumes P(i is 2nd | j won) = w_i / (1 - w_j). The discounted model
(Lo & Bacon-Shone) replaces the running weights with w_i**λ for placings after
1st. λ=1 recovers Harville; λ>1 suppresses longshots in lower placings, λ<1
amplifies them. We fit λ by maximizing the likelihood of the actual runner-up
across a sample of real races (win probs derived from final 単勝 odds).

Network: scrapes result pages for a sample of races. Run from repo root:
    python docs/analysis/fit_place_exponent.py
"""
import math
import time
from keiba.race_list import fetch_race_ids, fetch_kaisai_dates
from keiba.result_page import fetch_result_rows

MONTHS = [(2026, 5), (2026, 6), (2025, 11), (2025, 12)]
MAX_RACES = 500


def race_sample():
    dates = []
    for y, m in MONTHS:
        try:
            dates.extend(fetch_kaisai_dates(y, m))
        except Exception as exc:  # noqa: BLE001
            print(f"  ! {y}-{m} calendar: {exc}")
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


def race_records():
    """Yield (win_probs list, winner_idx, runnerup_idx) per usable race."""
    recs = []
    for rid in race_sample():
        try:
            rows = fetch_result_rows(rid)
        except Exception:  # noqa: BLE001
            continue
        time.sleep(0.25)
        horses = [(r["finish"], r["win_odds"]) for r in rows
                  if r.get("finish") and r.get("win_odds")]
        if len(horses) < 6:
            continue
        inv = [1.0 / o for _, o in horses]
        s = sum(inv)
        w = [x / s for x in inv]
        try:
            wi = next(i for i, (f, _) in enumerate(horses) if f == 1)
            ri = next(i for i, (f, _) in enumerate(horses) if f == 2)
        except StopIteration:
            continue
        recs.append((w, wi, ri))
    return recs


def loglik(recs, lam):
    ll = 0.0
    for w, wi, ri in recs:
        num = w[ri] ** lam
        denom = sum(w[k] ** lam for k in range(len(w)) if k != wi)
        if num <= 0 or denom <= 0:
            continue
        ll += math.log(num / denom)
    return ll


def main():
    recs = race_records()
    print(f"使用レース数: {len(recs)}\n")
    if len(recs) < 30:
        print("サンプルが少なすぎます。DATES を増やしてください。")
        return
    best_lam, best_ll = 1.0, loglik(recs, 1.0)
    print(f"{'lambda':>7} {'loglik':>12}")
    for i in range(5, 26):           # 0.5 .. 2.5
        lam = i / 10.0
        ll = loglik(recs, lam)
        mark = ""
        if ll > best_ll:
            best_ll, best_lam = ll, lam
        print(f"{lam:>7.1f} {ll:>12.1f}")
    print(f"\nλ=1 (Harville) LL = {loglik(recs, 1.0):.1f}")
    print(f"最良 λ* = {best_lam}  LL = {best_ll:.1f}")
    print("λ>1 なら『穴は2着に来にくい=Harvilleは穴の連対を過大評価』を意味する。")


if __name__ == "__main__":
    main()
