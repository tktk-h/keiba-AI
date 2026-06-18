"""スピード指数が『オッズを超える追加情報』を持つかの検証(再収集の前段)。

50レースの各出走馬について、レース日より前の過去走タイムからスピード指数を作り、
  勝ち ~ 市場勝率(logit) [+ スピード指数]
のロジスティック回帰を比較。スピード指数を足して log-loss が下がり、係数が
有意な大きさなら『オッズに無い情報を持つ』=本格組込みの価値あり、と判断する。

Usage: python validate_speed.py
"""
import numpy as np
import pandas as pd
from pathlib import Path
import time as _time
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from keiba.race_list import fetch_race_ids
from keiba.horse_page import fetch_results
from keiba.speed_figure import (build_baselines, condition_offsets,
                                horse_speed_rating)

ROWS_CACHE = "data_bets_rows_20260502_20260614.csv"
DATES = ["20260614", "20260613"]   # latest first
HIST_CACHE = "data_speed_hist.csv"  # horse_id,date,time,surface,distance,track_condition
N_RACES = 50


def race_date_map(dates):
    out = {}
    for d in dates:
        slash = f"{d[:4]}/{d[4:6]}/{d[6:]}"
        for rid in fetch_race_ids(d):
            out[rid] = slash
    return out


def load_histories(horse_ids):
    cached = {}
    if Path(HIST_CACHE).exists():
        df = pd.read_csv(HIST_CACHE, dtype={"horse_id": str})
        for hid, g in df.groupby("horse_id"):
            cached[hid] = g
    todo = [h for h in horse_ids if h not in cached]
    print(f"  履歴: キャッシュ{len(cached)}頭 / 新規{len(todo)}頭")
    recs = []
    for i, hid in enumerate(todo, 1):
        try:
            for r in fetch_results(hid, limit=30):
                recs.append({"horse_id": hid, "date": r.date, "time": r.time,
                             "surface": r.surface, "distance": r.distance,
                             "track_condition": r.track_condition})
        except Exception as exc:  # noqa: BLE001
            print(f"  ! {hid}: {exc}")
        if i % 100 == 0:
            print(f"    {i}/{len(todo)} ...")
        _time.sleep(0.8)
    if recs:
        new = pd.DataFrame(recs)
        if Path(HIST_CACHE).exists():
            new = pd.concat([pd.read_csv(HIST_CACHE, dtype={"horse_id": str}), new],
                            ignore_index=True)
        new.to_csv(HIST_CACHE, index=False)
    allc = pd.read_csv(HIST_CACHE, dtype={"horse_id": str})
    return {hid: g for hid, g in allc.groupby("horse_id")}


def main():
    raw = pd.read_csv(ROWS_CACHE, dtype={"horse_id": str})
    rdate = race_date_map(DATES)
    raw = raw[raw["race_id"].astype(str).isin(rdate)].copy()
    races = sorted(raw["race_id"].astype(str).unique(),
                   key=lambda x: (rdate[x], x), reverse=True)[:N_RACES]
    raw = raw[raw["race_id"].astype(str).isin(races)].copy()
    print(f"  対象: {len(races)}レース / {len(raw)}頭")

    hist = load_histories(sorted(raw["horse_id"].unique()))
    # baselines + going offsets from all fetched past runs
    corpus = pd.concat(hist.values(), ignore_index=True).to_dict("records")
    baselines = build_baselines(corpus, min_samples=5)
    offsets = condition_offsets(corpus)
    print(f"  baseline {len(baselines)}バケット / 馬場補正 "
          f"{ {k: round(v,2) for k,v in offsets.items()} }")

    # per-runner: market prob (within race) + pre-race speed rating
    raw["inv"] = 1.0 / raw["win_odds"].where(raw["win_odds"] > 0)
    raw["mkt"] = raw["inv"] / raw.groupby("race_id")["inv"].transform("sum")
    ratings = []
    for r in raw.itertuples():
        d = rdate[str(r.race_id)]
        runs = hist.get(str(r.horse_id))
        past = []
        if runs is not None:
            g = runs[runs["date"] < d].sort_values("date", ascending=False).head(5)
            past = list(g.itertuples())
        ratings.append(horse_speed_rating(past, baselines, offsets, n=5))
    raw["speed"] = ratings

    df = raw.dropna(subset=["mkt", "won"]).copy()
    cover = df["speed"].notna().mean()
    df["speed"] = df["speed"].fillna(50.0)  # neutral for no-history horses
    df["mkt_logit"] = np.log(df["mkt"].clip(1e-6, 1 - 1e-6) /
                             (1 - df["mkt"].clip(1e-6, 1 - 1e-6)))
    df["speed_z"] = (df["speed"] - df["speed"].mean()) / df["speed"].std(ddof=0)
    print(f"  スピード指数が付いた馬の割合: {cover:.0%}  サンプル {len(df)}")

    y = df["won"].astype(int).values
    Xm = df[["mkt_logit"]].values
    Xms = df[["mkt_logit", "speed_z"]].values

    def cv_logloss(X):
        m = LogisticRegression(max_iter=1000)
        return -cross_val_score(m, X, y, cv=5, scoring="neg_log_loss").mean()

    ll_m = cv_logloss(Xm)
    ll_ms = cv_logloss(Xms)
    coef = LogisticRegression(max_iter=1000).fit(Xms, y).coef_[0]
    print("\n=== 結果 ===")
    print(f"  log-loss  市場のみ      : {ll_m:.4f}")
    print(f"  log-loss  市場+スピード : {ll_ms:.4f}  (改善 {ll_m - ll_ms:+.4f})")
    print(f"  係数  market_logit={coef[0]:.3f}  speed_z={coef[1]:.3f}")
    verdict = "→ スピード指数はオッズに無い情報を持つ可能性あり" if ll_ms < ll_m - 1e-4 \
        else "→ オッズを超える追加情報は確認できず"
    print(" ", verdict)


if __name__ == "__main__":
    main()
