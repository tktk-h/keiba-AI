"""Simulate 回収率 (ROI) across multiple bet types on last weekend's real
JRA results: 単勝 / 複勝 / ワイド / 馬連.

A bet "hits" iff its set of 馬番 appears in that race's payout table for that
bet type (the payout table only lists winning combinations). So we don't need
finishing positions — membership in the payout table defines the hit and gives
the realized dividend (per 100円).

Selection (model side):
  単勝 = highest model win prob
  複勝 = highest model place prob
  ワイド/馬連 = the two highest model-win-prob horses (a pair)
Baseline (1番人気 side) mirrors each using lowest win_odds / its pair.

Usage:
    python sim_bets.py 20260613 20260614
"""
import sys
import glob
import time
import re
import pandas as pd
from pathlib import Path
import requests
from bs4 import BeautifulSoup

from keiba.race_list import fetch_race_ids
from keiba.result_page import parse_result_page
from keiba.dataset import build_dataset
from keiba.model import train_model, model_win_probabilities
from keiba.predictor import place_k, place_probabilities

HEADERS = {"User-Agent": "Mozilla/5.0 (keiba-research)"}

# payout-table th label -> our bet-type key
TYPE_LABELS = {"単勝": "win", "複勝": "place", "馬連": "quinella", "ワイド": "wide"}


def parse_payouts(html: str) -> dict:
    """{bet_type: {frozenset(馬番): 配当(per100円)}} from the result payout table.

    The combos count = number of dividends listed; we split the 馬番 cell's
    digits evenly into that many groups (robust to <br>/span quirks)."""
    soup = BeautifulSoup(html, "html.parser")
    out = {v: {} for v in TYPE_LABELS.values()}
    for t in soup.select("table.pay_table_01"):
        for tr in t.select("tr"):
            th = tr.select_one("th")
            if not th:
                continue
            key = TYPE_LABELS.get(th.get_text(strip=True))
            if not key:
                continue
            tds = tr.select("td")
            if len(tds) < 2:
                continue
            pays = [float(x.replace(",", "")) for x in
                    re.findall(r"[\d,]+", tds[1].get_text(" ", strip=True))]
            nums = [int(x) for x in re.findall(r"\d+", tds[0].get_text(" ", strip=True))]
            if not pays or not nums or len(nums) % len(pays) != 0:
                continue
            per = len(nums) // len(pays)
            for i, pay in enumerate(pays):
                combo = frozenset(nums[i * per:(i + 1) * per])
                out[key][combo] = pay
    return out


def fetch_result_and_payout(rid: str):
    resp = requests.get(f"https://db.netkeiba.com/race/{rid}/", headers=HEADERS, timeout=15)
    resp.raise_for_status()
    resp.encoding = "EUC-JP"
    return parse_result_page(resp.text, rid), parse_payouts(resp.text)


def load_train():
    paths = sorted(glob.glob("data_months/*_v3.csv"))
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    print(f"  学習データ: {len(paths)}ヶ月 / {df['race_id'].nunique()}レース / {len(df)}行")
    return df


def collect_weekend(dates):
    """Return (raw_rows, payouts{race_id:{type:{frozenset:pay}}}) with cache."""
    cache_rows = Path(f"data_bets_rows_{dates[0]}_{dates[-1]}.csv")
    cache_pay = Path(f"data_bets_pay_{dates[0]}_{dates[-1]}.csv")
    if cache_rows.exists() and cache_pay.exists():
        print(f"  キャッシュ利用: {cache_rows.name}")
        rows = pd.read_csv(cache_rows).to_dict("records")
        payouts = {}
        for _, r in pd.read_csv(cache_pay).iterrows():
            rid = str(int(r["race_id"]))
            combo = frozenset(int(x) for x in str(r["combo"]).split("-"))
            payouts.setdefault(rid, {}).setdefault(r["bet_type"], {})[combo] = float(r["pay"])
        return rows, payouts
    ids = sorted(set(i for d in dates for i in fetch_race_ids(d)))
    print(f"  先週末 {len(ids)} レースを収集中(時間がかかります)...")
    rows, payouts, pay_recs = [], {}, []
    for rid in ids:
        try:
            rr, pp = fetch_result_and_payout(rid)
            rows.extend(rr)
            payouts[rid] = pp
            for bt, combos in pp.items():
                for combo, pay in combos.items():
                    pay_recs.append({"race_id": rid, "bet_type": bt,
                                     "combo": "-".join(map(str, sorted(combo))), "pay": pay})
        except Exception as exc:  # noqa: BLE001
            print(f"  ! {rid} 取得失敗: {exc}")
        time.sleep(1.0)
    pd.DataFrame(rows).to_csv(cache_rows, index=False)
    pd.DataFrame(pay_recs).to_csv(cache_pay, index=False)
    print(f"  -> {cache_rows.name} ({len(rows)}行), {cache_pay.name} に保存")
    return rows, payouts


def simulate(feat, raw, payouts, model, side, stake=100):
    """side: 'model' or 'fav'. Returns ROI per bet type."""
    num_of = {(str(r["race_id"]), r["name"]): r["number"] for r in raw}
    types = ["win", "place", "wide", "quinella"]
    acc = {t: {"bets": 0, "hits": 0, "spent": 0.0, "returned": 0.0} for t in types}

    for rid, race in feat.groupby("race_id"):
        rid = str(rid)
        raw_race = [r for r in raw if str(r["race_id"]) == rid]
        field = len(raw_race)
        k = place_k(field)
        pay = payouts.get(rid, {})
        win_probs = model_win_probabilities(model, race)
        if side == "model":
            order = [n for n, _ in sorted(win_probs.items(), key=lambda kv: kv[1], reverse=True)]
            place_probs = place_probabilities(win_probs, k) if k else {}
            place_top = max(place_probs, key=place_probs.get) if place_probs else None
        else:  # favorite: by win_odds ascending
            valid = sorted((r for r in raw_race if r.get("win_odds")), key=lambda r: r["win_odds"])
            order = [r["name"] for r in valid]
            place_top = order[0] if order else None

        def num(name):
            return num_of.get((rid, name))

        # picks: each is a set of 馬番 to look up in the payout table
        picks = {}
        if order:
            picks["win"] = {num(order[0])}
        if place_top is not None:
            picks["place"] = {num(place_top)}
        if len(order) >= 2:
            pair = {num(order[0]), num(order[1])}
            picks["wide"] = pair
            picks["quinella"] = pair

        for t, sel in picks.items():
            if None in sel:
                continue
            acc[t]["bets"] += 1
            acc[t]["spent"] += stake
            combos = pay.get(t, {})
            if frozenset(sel) in combos:
                acc[t]["hits"] += 1
                acc[t]["returned"] += stake * combos[frozenset(sel)] / 100.0

    for t in types:
        a = acc[t]
        a["roi"] = a["returned"] / a["spent"] if a["spent"] else 0.0
    return acc


JP = {"win": "単勝", "place": "複勝", "wide": "ワイド", "quinella": "馬連"}


def report(side_label, acc):
    print(f"\n--- {side_label} ---")
    for t in ["win", "place", "wide", "quinella"]:
        a = acc[t]
        hit = (a["hits"] / a["bets"] * 100) if a["bets"] else 0.0
        print(f"  [{JP[t]}] 賭け{a['bets']}件 的中{a['hits']}件({hit:.1f}%) "
              f"投資{a['spent']:.0f}円 払戻{a['returned']:.0f}円 回収率{a['roi']:.1%}")


def main(dates):
    print("=== 学習 ===")
    model = train_model(load_train())
    print("=== 先週末データ ===")
    raw, payouts = collect_weekend(dates)
    feat = build_dataset(raw)
    print(f"  検証: {feat['race_id'].nunique()}レース / {len(feat)}行")
    print("\n=== シミュレーション(100円/レース・各券種) ===")
    report("モデル", simulate(feat, raw, payouts, model, "model"))
    report("1番人気(基準)", simulate(feat, raw, payouts, model, "fav"))


if __name__ == "__main__":
    dates = [a for a in sys.argv[1:] if not a.startswith("--")] or ["20260613", "20260614"]
    main(dates)
