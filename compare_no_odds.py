"""オッズ依存を下げたモデルの実験。

市場(オッズ)由来の特徴量を学習から外し、馬個別(馬体重・斤量・距離・馬場・
過去成績form 等)だけで勝率を予測するモデルを作り、ベースラインと比較する:
  1) 特徴量の重み(係数)
  2) 1レースの予想勝率が市場からどれだけ離れるか
  3) EVポートフォリオ回収率(377レース・キャッシュ利用)

Usage: python compare_no_odds.py [--min-prob 0.05]
"""
import sys
import numpy as np
import pandas as pd

from keiba.dataset import FEATURE_COLUMNS, build_dataset
from keiba.model import train_model, model_win_probabilities
from sim_bets import load_train, collect_weekend
from sim_ev import load_odds, simulate, report

MARKET_COLS = {"win_odds", "log_odds", "odds_rank_pct", "popularity",
               "prev_avg_log_odds", "prev_avg_popularity"}
NO_ODDS_COLS = [c for c in FEATURE_COLUMNS if c not in MARKET_COLS]
DATES = ["20260502", "20260614"]


def show_coef(model, title):
    clf = model.named_steps["clf"]
    cols = model.feature_columns_
    coef = clf.coef_[0]
    print(f"\n--- {title} 重み(絶対値上位8) ---")
    for i in np.argsort(-np.abs(coef))[:8]:
        print(f"  {cols[i]:<22}{coef[i]:>8.3f}")


def main(min_prob):
    print("=== 学習 ===")
    train = load_train()
    m_base = train_model(train)
    m_no = train_model(train, feature_columns=NO_ODDS_COLS)
    print(f"  ベースライン特徴量 {len(m_base.feature_columns_)} / オッズ抜き {len(m_no.feature_columns_)}")
    show_coef(m_no, "オッズ抜きモデル")

    raw, payouts = collect_weekend(DATES)
    feat = build_dataset(raw)
    odds_all = load_odds(DATES, [str(r) for r in feat["race_id"].unique()])

    # 1レースで市場との乖離を見る
    rid = str(feat["race_id"].iloc[0])
    race = feat[feat["race_id"].astype(str) == rid]
    rr = [r for r in raw if str(r["race_id"]) == rid]
    inv = {r["name"]: 1.0 / r["win_odds"] for r in rr if r["win_odds"]}
    z = sum(inv.values())
    mkt = {k: v / z for k, v in inv.items()}
    fin = {r["name"]: r["finish"] for r in rr}
    wp_b = model_win_probabilities(m_base, race)
    wp_n = model_win_probabilities(m_no, race)
    print(f"\n--- 例: race_id={rid} 勝率(%) ---")
    print(f"  {'市場':>6}{'基準':>7}{'オッズ抜き':>9}{'着順':>5}")
    for n, _ in sorted(mkt.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {mkt[n]*100:>5.1f}{wp_b.get(n,0)*100:>7.1f}{wp_n.get(n,0)*100:>8.1f}"
              f"{fin.get(n,0):>6}")

    print(f"\n=== EVポートフォリオ回収率 (min_prob={min_prob:.2%}) ===")
    for label, model in [("基準(オッズあり)", m_base), ("オッズ抜き", m_no)]:
        print(f"\n--- {label} ---")
        acc, rwb = simulate(feat, raw, payouts, odds_all, model, 0.0, min_prob)
        report(acc, rwb, feat["race_id"].nunique())


if __name__ == "__main__":
    rest = sys.argv[1:]
    min_prob = 0.05
    if "--min-prob" in rest:
        min_prob = float(rest[rest.index("--min-prob") + 1])
    main(min_prob)
