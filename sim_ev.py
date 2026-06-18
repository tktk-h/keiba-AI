"""EV-maximizing simulation on last weekend's real JRA results.

For every race we enumerate ALL candidate bets across 単勝/複勝/ワイド/馬連,
compute EV = model_prob * pre-race_odds, and bet every candidate whose EV is
non-negative (+EV). Multiple bet types per race are allowed — the +EV set is
exactly the portfolio that maximizes expected value at unit stakes.

Pre-race odds come from the netkeiba odds API (all horses / all pairs).
Realized returns come from the result payout table (winning combos only).

Reuses caches written by sim_bets.py (raw rows + payouts) and adds an odds
cache of its own.

Usage:
    python sim_ev.py 20260613 20260614 [--min-ev 0.0]
"""
import sys
import glob
import time
import pandas as pd
from pathlib import Path

from keiba.odds_page import fetch_odds
from keiba.dataset import build_dataset
from keiba.model import train_model, model_win_probabilities
from keiba.predictor import (place_k, place_probabilities,
                             quinella_probabilities, wide_probabilities)
from keiba.expected_value import ev as ev_of
from sim_bets import collect_weekend, load_train, JP


def load_odds(dates, race_ids):
    """{race_id: odds dict} with a disk cache (odds API, ~3 calls/race)."""
    cache = Path(f"data_ev_odds_{dates[0]}_{dates[-1]}.csv")
    if cache.exists():
        print(f"  オッズ キャッシュ利用: {cache.name}")
        df = pd.read_csv(cache)
        odds = {}
        for _, r in df.iterrows():
            rid = str(int(r["race_id"]))
            d = odds.setdefault(rid, {"win": {}, "place": {}, "quinella": {}, "wide": {}})
            nums = tuple(int(x) for x in str(r["combo"]).split("-"))
            if r["bet_type"] in ("win", "place"):
                d[r["bet_type"]][nums[0]] = float(r["odds"])
            else:
                d[r["bet_type"]][nums] = float(r["odds"])
        return odds
    print(f"  {len(race_ids)} レースのオッズを取得中(時間がかかります)...")
    odds, recs = {}, []
    for rid in race_ids:
        try:
            o = fetch_odds(rid)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! {rid} オッズ取得失敗: {exc}")
            continue
        odds[rid] = o
        for num, v in o["win"].items():
            recs.append({"race_id": rid, "bet_type": "win", "combo": num, "odds": v})
        for num, (lo, hi) in o["place"].items():
            recs.append({"race_id": rid, "bet_type": "place", "combo": num, "odds": lo})
        for (a, b), v in o["quinella"].items():
            recs.append({"race_id": rid, "bet_type": "quinella", "combo": f"{a}-{b}", "odds": v})
        for (a, b), v in o["wide"].items():
            recs.append({"race_id": rid, "bet_type": "wide", "combo": f"{a}-{b}", "odds": v})
        time.sleep(0.8)
    pd.DataFrame(recs).to_csv(cache, index=False)
    print(f"  -> {cache.name} に保存")
    return odds


def candidate_bets(race_feat, raw_race, odds, model):
    """List of (bet_type, frozenset(numbers), prob, odds) for one race."""
    name2num = {r["name"]: r["number"] for r in raw_race}
    num2name = {r["number"]: r["name"] for r in raw_race}
    win_probs = model_win_probabilities(model, race_feat)
    k = place_k(len(raw_race))
    place_probs = place_probabilities(win_probs, k) if k else {}
    quin_probs = quinella_probabilities(win_probs)
    wide_probs = wide_probabilities(win_probs)

    out = []
    for name, p in win_probs.items():
        num = name2num.get(name)
        o = odds.get("win", {}).get(num)
        if num and o:
            out.append(("win", frozenset({num}), p, o))
    for name, p in place_probs.items():
        num = name2num.get(name)
        o = odds.get("place", {}).get(num)
        if num and o:
            lo = o[0] if isinstance(o, (tuple, list)) else o  # use 複勝下限
            out.append(("place", frozenset({num}), p, lo))
    for bt, probs, omap in (("quinella", quin_probs, odds.get("quinella", {})),
                            ("wide", wide_probs, odds.get("wide", {}))):
        for (a, b), o in omap.items():
            na, nb = num2name.get(a), num2name.get(b)
            if na is None or nb is None:
                continue
            p = probs.get(tuple(sorted((na, nb))), 0.0)
            if p > 0:
                out.append((bt, frozenset({a, b}), p, o))
    return out


def simulate(feat, raw, payouts, odds_all, model, min_ev=0.0, min_prob=0.0, stake=100):
    types = ["win", "place", "wide", "quinella"]
    acc = {t: {"bets": 0, "hits": 0, "spent": 0.0, "returned": 0.0} for t in types}
    races_with_bet = 0
    for rid, race in feat.groupby("race_id"):
        rid = str(rid)
        raw_race = [r for r in raw if str(r["race_id"]) == rid]
        odds = odds_all.get(rid)
        if not odds:
            continue
        cands = candidate_bets(race, raw_race, odds, model)
        pay = payouts.get(rid, {})
        bet_here = False
        for bt, combo, p, o in cands:
            if p < min_prob or ev_of(p, o) < min_ev:
                continue
            bet_here = True
            acc[bt]["bets"] += 1
            acc[bt]["spent"] += stake
            if combo in pay.get(bt, {}):
                acc[bt]["hits"] += 1
                acc[bt]["returned"] += stake * pay[bt][combo] / 100.0
        races_with_bet += 1 if bet_here else 0
    for t in types:
        a = acc[t]
        a["roi"] = a["returned"] / a["spent"] if a["spent"] else 0.0
    return acc, races_with_bet


def report(acc, races_with_bet, n_races):
    tot_bets = sum(a["bets"] for a in acc.values())
    tot_spent = sum(a["spent"] for a in acc.values())
    tot_ret = sum(a["returned"] for a in acc.values())
    print(f"\n  賭けたレース: {races_with_bet}/{n_races}  総買い目: {tot_bets}件")
    for t in ["win", "place", "wide", "quinella"]:
        a = acc[t]
        if not a["bets"]:
            print(f"  [{JP[t]}] 買い目なし(+EVなし)")
            continue
        hit = a["hits"] / a["bets"] * 100
        print(f"  [{JP[t]}] {a['bets']}件 的中{a['hits']}({hit:.1f}%) "
              f"投資{a['spent']:.0f} 払戻{a['returned']:.0f} 回収率{a['roi']:.1%}")
    tot_roi = tot_ret / tot_spent if tot_spent else 0.0
    print(f"  [合計] {tot_bets}件 投資{tot_spent:.0f} 払戻{tot_ret:.0f} 回収率{tot_roi:.1%}")


def main(dates, min_ev, min_prob):
    print("=== 学習 ===")
    model = train_model(load_train())
    print("=== 先週末データ ===")
    raw, payouts = collect_weekend(dates)
    feat = build_dataset(raw)
    n_races = feat["race_id"].nunique()
    print(f"  検証: {n_races}レース / {len(feat)}行")
    race_ids = [str(r) for r in sorted(feat["race_id"].unique())]
    odds_all = load_odds(dates, race_ids)
    print(f"\n=== EVポートフォリオ(各レースで EV>={min_ev:.1f} かつ 確率>={min_prob:.2%} を全部買う・100円/点) ===")
    acc, rwb = simulate(feat, raw, payouts, odds_all, model, min_ev=min_ev, min_prob=min_prob)
    report(acc, rwb, n_races)


if __name__ == "__main__":
    rest = sys.argv[1:]
    min_ev, min_prob = 0.0, 0.0
    for flag in ("--min-ev", "--min-prob"):
        if flag in rest:
            i = rest.index(flag)
            val = float(rest[i + 1])
            rest = rest[:i] + rest[i + 2:]
            if flag == "--min-ev":
                min_ev = val
            else:
                min_prob = val
    dates = [a for a in rest if not a.startswith("--")] or ["20260613", "20260614"]
    main(dates, min_ev, min_prob)
