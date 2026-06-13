"""Backtest the win model on collected race data and report 回収率 (ROI).

Usage:
    python backtest.py --month 2024-01 2024-06   # collect & backtest a range
    python backtest.py --csv data.csv            # backtest a saved dataset
    python backtest.py --month 2024-05 --save data.csv   # also save dataset

Splits races chronologically (older=train, newer=test) and compares the
model's bets against an "always bet the favorite" baseline.
ROI 1.0 = break-even; above 1.0 = profitable.
"""
import sys
import pandas as pd
from keiba.dataset import collect_dataset
from keiba.race_list import collect_race_ids
from keiba.backtest import run_backtest


def _months_in_range(start, end):
    sy, sm = (int(x) for x in start.split("-"))
    ey, em = (int(x) for x in end.split("-"))
    y, m = sy, sm
    while (y, m) <= (ey, em):
        yield y, m
        m += 1
        if m > 12:
            y, m = y + 1, 1


def _load_data(args):
    if "--csv" in args:
        return pd.read_csv(args[args.index("--csv") + 1])
    if "--month" in args:
        months = [a for a in args[args.index("--month") + 1:]
                  if not a.startswith("--")]
        start = months[0]
        end = months[1] if len(months) > 1 else months[0]
        ids = []
        for year, month in _months_in_range(start, end):
            print(f"  {year}-{month:02d} のレースID収集中...")
            ids.extend(collect_race_ids(year, month))
        ids = sorted(set(ids))
        print(f"  {len(ids)} レースの結果を収集中(時間がかかります)...")
        return collect_dataset(ids)
    return None


def _report(label, r):
    print(f"  [{label}] 賭け{r['bets']}件 的中{r['hits']}件 "
          f"投資{r['spent']:.0f}円 払戻{r['returned']:.0f}円 "
          f"回収率{r['roi']:.1%}")


def main(args):
    df = _load_data(args)
    if df is None or df.empty:
        print("データがありません。--month か --csv を指定してください。")
        return
    if "--save" in args:
        path = args[args.index("--save") + 1]
        df.to_csv(path, index=False)
        print(f"  データセットを {path} に保存しました({len(df)}行)。")

    result = run_backtest(df)
    print(f"\n=== バックテスト結果 ===")
    print(f"学習 {result['train_races']} レース / 検証 {result['test_races']} レース")
    _report("モデル", result["model"])
    _report("1番人気", result["favorite"])
    diff = result["model"]["roi"] - result["favorite"]["roi"]
    verdict = "モデルが上回りました" if diff > 0 else "1番人気が上回りました"
    print(f"差: {diff:+.1%} → {verdict}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1:])
