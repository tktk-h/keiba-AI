"""enrich版: 各馬の『そのレース日より前の直近5走』から form 特徴量を作り、
form あり/なし を同じレース集合で比較する(50レースのサンプル)。

本番(build_race_report --enrich)と同じく horse 結果ページから過去走を取得。
ただし過去レースの検証なので、評価レース日以降の走りは look-ahead として除外する。
特徴量は build_dataset の出力(BASE+RELATIVE+FORM)を土台に、FORM列だけを
履歴ベースの正しい値で上書きする。オッズ/払戻は既存キャッシュを再利用。

Usage:
    python sim_ev_enrich.py            # 直近50レース・min_prob 0.05
    python sim_ev_enrich.py --n 50 --min-prob 0.05
"""
import sys
import time
import pandas as pd
from pathlib import Path

from keiba.race_list import fetch_race_ids
from keiba.horse_page import fetch_results
from keiba.features import _form_from_past_runs
from keiba.form_features import FORM_COLUMNS
from keiba.dataset import build_dataset
from keiba.model import train_model
from sim_bets import load_train
from sim_ev import load_odds, simulate, report

DATES = ["20260614", "20260613", "20260607", "20260606"]  # latest first
ROWS_CACHE = "data_bets_rows_20260502_20260614.csv"
PAY_CACHE = "data_bets_pay_20260502_20260614.csv"
ODDS_DATES = ["20260502", "20260614"]  # cache key span for load_odds


def race_date_map(needed_dates):
    """{race_id: 'YYYY/MM/DD'} for the given kaisai dates (network, small)."""
    out = {}
    for d in needed_dates:
        slash = f"{d[:4]}/{d[4:6]}/{d[6:]}"
        for rid in fetch_race_ids(d):
            out[rid] = slash
    return out


def load_histories(horse_ids):
    """{horse_id: [PastRun ...] newest-first} with a disk cache."""
    cache = Path("data_enrich_hist.csv")
    cached = {}
    if cache.exists():
        df = pd.read_csv(cache, dtype={"horse_id": str})
        for hid, g in df.groupby("horse_id"):
            cached[hid] = g
    out, recs = {}, []
    todo = [h for h in horse_ids if h not in cached]
    print(f"  履歴: キャッシュ{len(cached)}頭 / 新規取得{len(todo)}頭")
    for i, hid in enumerate(todo, 1):
        try:
            runs = fetch_results(hid, limit=50)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! {hid} 取得失敗: {exc}")
            runs = []
        for r in runs:
            recs.append({"horse_id": hid, "date": r.date, "finish": r.finish,
                         "win_odds": r.win_odds, "popularity": r.popularity,
                         "last_3f": r.last_3f})
        if i % 50 == 0:
            print(f"    {i}/{len(todo)} ...")
        time.sleep(0.8)
    if recs:
        new = pd.DataFrame(recs)
        if cache.exists():
            new = pd.concat([pd.read_csv(cache, dtype={"horse_id": str}), new], ignore_index=True)
        new.to_csv(cache, index=False)
    # reload everything as grouped frames
    allcache = pd.read_csv(cache, dtype={"horse_id": str})
    return {hid: g for hid, g in allcache.groupby("horse_id")}


class _PR:  # lightweight PastRun for _form_from_past_runs
    __slots__ = ("finish", "win_odds", "last_3f", "popularity")

    def __init__(self, finish, win_odds, last_3f, popularity):
        self.finish = finish
        self.win_odds = win_odds
        self.last_3f = last_3f
        self.popularity = popularity


def past_runs_before(hist_df, race_date, limit=5):
    if hist_df is None:
        return []
    g = hist_df[hist_df["date"] < race_date].copy()
    g = g.sort_values("date", ascending=False).head(limit)
    return [_PR(r.finish, r.win_odds, r.last_3f, r.popularity)
            for r in g.itertuples()]


def apply_form(feat, raw, histories, rdate):
    """Return a copy of feat with FORM_COLUMNS overwritten by historical form."""
    hid_of = {(str(r["race_id"]), r["name"]): str(r["horse_id"]) for r in raw}
    out = feat.copy()
    for rid, race in feat.groupby("race_id"):
        rid = str(rid)
        d = rdate.get(rid)
        if d is None:
            continue
        # field-wide last_3f fill from the selected past runs (mirror build_features)
        all_l3, per_horse = [], {}
        for name in race["name"]:
            hist = histories.get(hid_of.get((rid, name)))
            runs = past_runs_before(hist, d)
            per_horse[name] = runs
            all_l3 += [pr.last_3f for pr in runs if pr.last_3f is not None]
        fill = sum(all_l3) / len(all_l3) if all_l3 else 0.0
        for idx, name in zip(race.index, race["name"]):
            form = _form_from_past_runs(per_horse[name], fill)
            for c in FORM_COLUMNS:
                out.at[idx, c] = form[c]
    return out


def main(n, min_ev, min_prob):
    print("=== 学習 ===")
    model = train_model(load_train())
    print("=== データ(キャッシュ) ===")
    raw_all = pd.read_csv(ROWS_CACHE, dtype={"horse_id": str}).to_dict("records")
    payouts = {}
    for _, r in pd.read_csv(PAY_CACHE).iterrows():
        rid = str(int(r["race_id"]))
        combo = frozenset(int(x) for x in str(r["combo"]).split("-"))
        payouts.setdefault(rid, {}).setdefault(r["bet_type"], {})[combo] = float(r["pay"])

    print("  レース日マップ取得中...")
    rdate = race_date_map(DATES)
    # subset: latest races present in the data, up to n
    feat_all = build_dataset(raw_all)
    avail = [rid for rid in sorted(feat_all["race_id"].astype(str).unique(),
                                   key=lambda x: (rdate.get(x, ""), x), reverse=True)
             if rid in rdate][:n]
    avail = set(avail)
    raw = [r for r in raw_all if str(r["race_id"]) in avail]
    feat_off = build_dataset(raw)
    print(f"  対象: {feat_off['race_id'].nunique()}レース / {len(feat_off)}行")

    hids = sorted({str(r["horse_id"]) for r in raw})
    histories = load_histories(hids)
    feat_on = apply_form(feat_off, raw, histories, rdate)

    odds_all = load_odds(ODDS_DATES, [str(r) for r in feat_off["race_id"].unique()])

    print(f"\n=== form なし(週末内計算) min_prob={min_prob:.2%} ===")
    acc, rwb = simulate(feat_off, raw, payouts, odds_all, model, min_ev, min_prob)
    report(acc, rwb, feat_off["race_id"].nunique())
    print(f"\n=== form あり(履歴・直近5走) min_prob={min_prob:.2%} ===")
    acc, rwb = simulate(feat_on, raw, payouts, odds_all, model, min_ev, min_prob)
    report(acc, rwb, feat_on["race_id"].nunique())


if __name__ == "__main__":
    rest = sys.argv[1:]
    n, min_ev, min_prob = 50, 0.0, 0.05
    for flag, conv in (("--n", int), ("--min-ev", float), ("--min-prob", float)):
        if flag in rest:
            i = rest.index(flag)
            val = conv(rest[i + 1])
            rest = rest[:i] + rest[i + 2:]
            if flag == "--n":
                n = val
            elif flag == "--min-ev":
                min_ev = val
            else:
                min_prob = val
    main(n, min_ev, min_prob)
