"""Simulate 複勝 (place) 回収率 on last weekend's real JRA results.

Win bets only need win_odds, but 複勝 needs (a) each horse's finishing
position and (b) the actual 複勝 dividend from the result page's payout table.
We fetch each result page once and parse both the result rows and the payout
table, then bet 複勝 on the model's top pick per race and tally ROI.

Strategy compared:
  - モデル: bet 複勝 on the horse with the highest model place-probability.
  - 1番人気: bet 複勝 on the lowest-odds (most popular) horse.
A 100円 複勝 bet returns the dividend (per-100円) when the horse finishes
within the place cutoff (3着 for 8+ runners, 2着 for 5-7, none for <=4).

Usage:
    python sim_place.py 20260613 20260614
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


def parse_place_payouts(html: str) -> dict:
    """{馬番(int): 複勝払戻(per 100円, float)} from the result payout table."""
    soup = BeautifulSoup(html, "html.parser")
    for t in soup.select("table.pay_table_01"):
        for tr in t.select("tr"):
            th = tr.select_one("th")
            if not th or "複勝" not in th.get_text(strip=True):
                continue
            tds = tr.select("td")
            if len(tds) < 2:
                continue
            nums = [int(x) for x in re.findall(r"\d+", tds[0].get_text("\n", strip=True).replace("\n", " "))]
            pays = [float(x.replace(",", "")) for x in tds[1].get_text("\n", strip=True).split("\n") if x.strip()]
            return {n: p for n, p in zip(nums, pays)}
    return {}


def fetch_result_and_payout(rid: str):
    resp = requests.get(f"https://db.netkeiba.com/race/{rid}/", headers=HEADERS, timeout=15)
    resp.raise_for_status()
    resp.encoding = "EUC-JP"
    return parse_result_page(resp.text, rid), parse_place_payouts(resp.text)


def load_train():
    paths = sorted(glob.glob("data_months/*_v3.csv"))
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    print(f"  学習データ: {len(paths)}ヶ月 / {df['race_id'].nunique()}レース / {len(df)}行")
    return df


def collect_weekend(dates):
    """Return (raw_rows list, payouts {race_id:{num:pay}}) with on-disk cache."""
    cache_rows = Path(f"data_place_rows_{dates[0]}_{dates[-1]}.csv")
    cache_pay = Path(f"data_place_pay_{dates[0]}_{dates[-1]}.csv")
    if cache_rows.exists() and cache_pay.exists():
        print(f"  キャッシュ利用: {cache_rows.name}")
        rows = pd.read_csv(cache_rows).to_dict("records")
        paydf = pd.read_csv(cache_pay)
        payouts = {}
        for _, r in paydf.iterrows():
            payouts.setdefault(str(int(r["race_id"])), {})[int(r["number"])] = float(r["pay"])
        return rows, payouts
    ids = sorted(set(i for d in dates for i in fetch_race_ids(d)))
    print(f"  先週末 {len(ids)} レースを収集中(時間がかかります)...")
    rows, payouts, pay_records = [], {}, []
    for rid in ids:
        try:
            rr, pp = fetch_result_and_payout(rid)
            rows.extend(rr)
            payouts[rid] = pp
            for num, pay in pp.items():
                pay_records.append({"race_id": rid, "number": num, "pay": pay})
        except Exception as exc:  # noqa: BLE001
            print(f"  ! {rid} 取得失敗: {exc}")
        time.sleep(1.0)
    pd.DataFrame(rows).to_csv(cache_rows, index=False)
    pd.DataFrame(pay_records).to_csv(cache_pay, index=False)
    print(f"  -> {cache_rows.name} ({len(rows)}行), {cache_pay.name} に保存")
    return rows, payouts


def simulate(feat, raw, payouts, model, picker, stake=100):
    """picker(race_feat_df, win_probs, place_probs, raw_race_df) -> horse name."""
    spent = returned = 0.0
    bets = hits = 0
    finish = {(str(r["race_id"]), r["name"]): r["finish"] for r in raw}
    num_of = {(str(r["race_id"]), r["name"]): r["number"] for r in raw}
    for rid, race in feat.groupby("race_id"):
        rid = str(rid)
        raw_race = [r for r in raw if str(r["race_id"]) == rid]
        field = len(raw_race)
        k = place_k(field)
        if k is None:
            continue
        win_probs = model_win_probabilities(model, race)
        place_probs = place_probabilities(win_probs, k)
        name = picker(race, win_probs, place_probs, raw_race)
        if name is None:
            continue
        bets += 1
        spent += stake
        fin = finish.get((rid, name))
        num = num_of.get((rid, name))
        if fin is not None and fin <= k and num in payouts.get(rid, {}):
            hits += 1
            returned += stake * payouts[rid][num] / 100.0
    roi = returned / spent if spent else 0.0
    return {"bets": bets, "hits": hits, "spent": spent, "returned": returned, "roi": roi}


def report(label, r):
    hit = (r["hits"] / r["bets"] * 100) if r["bets"] else 0.0
    print(f"  [{label}] 賭け{r['bets']}件 的中{r['hits']}件({hit:.1f}%) "
          f"投資{r['spent']:.0f}円 払戻{r['returned']:.0f}円 回収率{r['roi']:.1%}")


def main(dates):
    print("=== 学習 ===")
    model = train_model(load_train())
    print("=== 先週末データ ===")
    raw, payouts = collect_weekend(dates)
    feat = build_dataset(raw)
    print(f"  検証: {feat['race_id'].nunique()}レース / {len(feat)}行")

    def model_pick(race, win_probs, place_probs, raw_race):
        if not place_probs:
            return None
        return max(place_probs, key=place_probs.get)

    def fav_pick(race, win_probs, place_probs, raw_race):
        valid = [r for r in raw_race if r.get("win_odds")]
        if not valid:
            return None
        return min(valid, key=lambda r: r["win_odds"])["name"]

    print("\n=== 複勝シミュレーション(100円/レース) ===")
    report("モデル(複勝率最上位)", simulate(feat, raw, payouts, model, model_pick))
    report("1番人気(基準)", simulate(feat, raw, payouts, model, fav_pick))


if __name__ == "__main__":
    dates = [a for a in sys.argv[1:] if not a.startswith("--")] or ["20260613", "20260614"]
    main(dates)
