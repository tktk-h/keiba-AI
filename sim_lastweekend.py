"""Simulate current 回収率 (ROI) on last weekend's real JRA results.

Trains the win model on the full v3 history (data_months/*_v3.csv) and
evaluates the EV-based picks against the actual results of the given
weekend dates. ROI 1.0 = break-even; >1.0 = profit.

Usage:
    python sim_lastweekend.py 20260613 20260614
"""
import sys
import glob
import pandas as pd
from pathlib import Path

from keiba.race_list import fetch_race_ids
from keiba.dataset import collect_dataset
from keiba.model import train_model
from keiba.backtest import simulate_win_bets, model_picks, favorite_picks


def load_train():
    paths = sorted(glob.glob("data_months/*_v3.csv"))
    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    print(f"  学習データ: {len(paths)}ヶ月 / {df['race_id'].nunique()}レース / {len(df)}行")
    return df


def load_test(dates):
    cache = Path(f"data_lastweekend_{dates[0]}_{dates[-1]}.csv")
    if cache.exists():
        print(f"  キャッシュ利用: {cache}")
        return pd.read_csv(cache)
    ids = []
    for d in dates:
        ids.extend(fetch_race_ids(d))
    ids = sorted(set(ids))
    print(f"  先週末 {len(ids)} レースの結果を収集中(時間がかかります)...")
    df = collect_dataset(ids)
    df.to_csv(cache, index=False)
    print(f"  -> {cache} に保存({len(df)}行)")
    return df


def report(label, r):
    hit = (r["hits"] / r["bets"] * 100) if r["bets"] else 0.0
    print(f"  [{label}] 賭け{r['bets']}件 的中{r['hits']}件({hit:.1f}%) "
          f"投資{r['spent']:.0f}円 払戻{r['returned']:.0f}円 "
          f"回収率{r['roi']:.1%}")


def main(dates):
    print("=== 学習 ===")
    train = load_train()
    model = train_model(train)
    print("=== 先週末データ ===")
    test = load_test(dates)
    print(f"  検証: {test['race_id'].nunique()}レース / {len(test)}行")

    print("\n=== 単勝シミュレーション(100円/レース) ===")
    # min_ev=1.0: skip -EV races. Also show min_ev=0 (always bet best EV horse).
    for label, mev in [("モデル(EV>=1.0のみ)", 1.0), ("モデル(全レース)", 0.0)]:
        picks = model_picks(model, test, min_ev=mev)
        report(label, simulate_win_bets(test, picks))
    report("1番人気(基準)", simulate_win_bets(test, favorite_picks(test)))


if __name__ == "__main__":
    dates = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not dates:
        dates = ["20260613", "20260614"]
    main(dates)
