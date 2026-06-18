"""既存特徴量のまま『モデル/較正/選択』を変えて回収率がどこまで動くか比較。

キャッシュ済み377レース(2026-05..06-14)のEVポートフォリオで、
  - ロジスティック回帰(現行)
  - LightGBM(非線形・特徴量の相互作用を拾う)
を、確率フロア(min_prob)を振って合計回収率で比べる。ネット不要。

Usage: python ev_model_compare.py
"""
import pandas as pd
from keiba.dataset import build_dataset
from keiba.model import train_model, train_lgbm_model
from sim_bets import load_train, collect_weekend
from sim_ev import load_odds, simulate

DATES = ["20260502", "20260614"]
FLOORS = [0.0, 0.03, 0.05, 0.10]


def total_roi(acc):
    spent = sum(a["spent"] for a in acc.values())
    ret = sum(a["returned"] for a in acc.values())
    bets = sum(a["bets"] for a in acc.values())
    return bets, (ret / spent if spent else 0.0)


def main():
    print("=== 学習 ===")
    train = load_train()
    models = {"ロジスティック": train_model(train),
              "LightGBM": train_lgbm_model(train)}
    raw, payouts = collect_weekend(DATES)
    feat = build_dataset(raw)
    odds_all = load_odds(DATES, [str(r) for r in feat["race_id"].unique()])
    print(f"  検証 {feat['race_id'].nunique()}レース\n")

    header = "モデル".ljust(14) + "".join(f"min_prob={p:>4.0%}".rjust(16) for p in FLOORS)
    print(header)
    for name, model in models.items():
        cells = []
        for p in FLOORS:
            acc, _ = simulate(feat, raw, payouts, odds_all, model, 0.0, p)
            bets, roi = total_roi(acc)
            cells.append(f"{roi:>6.1%}({bets})".rjust(16))
        print(name.ljust(14) + "".join(cells))
    print("\n※ (n)=総買い目数。回収率は控除率の壁(約80%)が上限の目安。")


if __name__ == "__main__":
    main()
