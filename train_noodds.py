"""独自予想モデル(model_noodds.pkl)を学習・保存する。

市場(オッズ)由来の特徴量を全て除外し、馬の中身(近走着順・上がり・出走数・勝率・
馬体重・斤量・距離・芝ダ・馬場・頭数 等)だけで勝率を出す logistic モデル。
ウォークフォワード比較(odds_free_tune.py)で「オッズ抜きで最も精度が高く、かつ
最も独自」の構成として全市場除外×logisticを採用。
"""
import glob
import pandas as pd
from keiba.dataset import FEATURE_COLUMNS
from keiba.model import train_model, save_model

MARKET = {"win_odds", "log_odds", "odds_rank_pct", "popularity",
          "prev_avg_log_odds", "prev_avg_popularity"}
NOODDS_COLUMNS = [c for c in FEATURE_COLUMNS if c not in MARKET]


def main():
    paths = sorted(glob.glob("data_months/*_v3.csv"))
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    model = train_model(df, feature_columns=NOODDS_COLUMNS)
    save_model(model, "model_noodds.pkl")
    print(f"  学習 {df['race_id'].nunique()}レース / {len(df)}行")
    print(f"  -> model_noodds.pkl 保存。特徴量({len(NOODDS_COLUMNS)}):")
    print("   ", NOODDS_COLUMNS)


if __name__ == "__main__":
    main()
