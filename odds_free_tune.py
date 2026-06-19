"""オッズを抜いた中で精度を最大化する構成を選ぶ(ウォークフォワード)。

精度指標(予想勝率の良さ):
  - log-loss: 実際の勝ち馬に与えた確率の -ln 平均(低いほど良い)
  - top1: モデルの最有力馬が実際に勝った割合(高いほど良い)

比較:
  withOdds   : 全特徴量(市場込み=精度の天井・参考)
  A_noCurOdds: 現在オッズ系のみ除外(win_odds/log_odds/odds_rank_pct/popularity)。
               過去の人気/オッズ(prev_avg_*)は能力情報として残す=独自だが精度寄り。
  B_noMarket : 市場由来を全部除外(prev_avg_log_odds/popularity も)=最も独自。
各セットで logistic と LightGBM を比較。

Usage: python odds_free_tune.py
"""
import glob
import re
import numpy as np
import pandas as pd
from keiba.dataset import FEATURE_COLUMNS
from keiba.model import train_model, train_lgbm_model, model_win_probabilities

CUR = {"win_odds", "log_odds", "odds_rank_pct", "popularity"}
MKT = CUR | {"prev_avg_log_odds", "prev_avg_popularity"}
SETS = {
    "withOdds": list(FEATURE_COLUMNS),
    "A_noCurOdds": [c for c in FEATURE_COLUMNS if c not in CUR],
    "B_noMarket": [c for c in FEATURE_COLUMNS if c not in MKT],
}
MIN_TRAIN = 12


def evaluate(model, test):
    ll, n, hit, races = 0.0, 0, 0, 0
    for _, race in test.groupby("race_id"):
        probs = model_win_probabilities(model, race)
        w = race.loc[race["won"] == 1, "name"]
        if w.empty or not probs:
            continue
        w = w.iloc[0]
        races += 1
        n += 1
        ll += -np.log(max(probs.get(w, 1e-6), 1e-6))
        if max(probs, key=probs.get) == w:
            hit += 1
    return ll / n, hit / races


def main():
    files = sorted(glob.glob("data_months/*_v3.csv"))
    months = [re.search(r"(\d{4}_\d{2})_v3", f).group(1) for f in files]
    frames = {m: pd.read_csv(f) for f, m in zip(files, months)}
    print(f"  月数 {len(files)} / テスト窓 {len(files) - MIN_TRAIN}")
    acc = {name: {"log": [], "lgb": []} for name in SETS}
    for i in range(MIN_TRAIN, len(files)):
        train = pd.concat([frames[m] for m in months[:i]], ignore_index=True)
        test = frames[months[i]]
        for name, cols in SETS.items():
            for tag, trainer in (("log", train_model), ("lgb", train_lgbm_model)):
                m = trainer(train, feature_columns=cols)
                acc[name][tag].append(evaluate(m, test))
    print("\n=== ウォークフォワード平均(全テスト月) ===")
    print(f"{'構成':14}{'モデル':6}{'log-loss':>10}{'top1的中':>10}")
    for name in SETS:
        for tag in ("log", "lgb"):
            arr = acc[name][tag]
            ll = np.mean([a for a, _ in arr])
            hit = np.mean([b for _, b in arr])
            print(f"{name:14}{tag:6}{ll:>10.3f}{hit:>9.1%}")
    print("\n※ withOdds が天井。A/B のうち top1 が高く log-loss が低い構成が"
          "『オッズ抜きで最も精度が高い』。")


if __name__ == "__main__":
    main()
