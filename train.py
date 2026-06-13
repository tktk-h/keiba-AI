"""Collect past race results from netkeiba and train the win model.

Usage:
    python train.py <race_id> [<race_id> ...]
    python train.py --file race_ids.txt       # one race_id per line
    python train.py --month 2024-05            # all races that month
    python train.py --month 2024-01 2024-12    # a range of months (inclusive)

Saves the trained model to model.pkl. Run main.py afterwards to use it.
The more races you collect, the better the model — aim for hundreds.
"""
import sys
from keiba.dataset import collect_dataset
from keiba.model import train_model, save_model, DEFAULT_MODEL_PATH
from keiba.race_list import collect_race_ids


def _months_in_range(start, end):
    sy, sm = (int(x) for x in start.split("-"))
    ey, em = (int(x) for x in end.split("-"))
    y, m = sy, sm
    while (y, m) <= (ey, em):
        yield y, m
        m += 1
        if m > 12:
            y, m = y + 1, 1


def _read_ids(args):
    if "--month" in args:
        months = [a for a in args[args.index("--month") + 1:]
                  if not a.startswith("--")]
        if not months:
            return []
        start = months[0]
        end = months[1] if len(months) > 1 else months[0]
        ids = []
        for year, month in _months_in_range(start, end):
            print(f"  {year}-{month:02d} の開催レースID収集中...")
            ids.extend(collect_race_ids(year, month))
        return sorted(set(ids))
    if "--file" in args:
        path = args[args.index("--file") + 1]
        with open(path, encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    return [a for a in args if not a.startswith("--")]


def main(race_ids):
    print(f"{len(race_ids)} レースの結果を収集中...")
    df = collect_dataset(race_ids)
    if df.empty:
        print("データが集まりませんでした。レースIDを確認してください。")
        return
    wins = int(df["won"].sum())
    print(f"  学習データ: {len(df)} 頭分 / 勝ち {wins} 件")
    model = train_model(df)
    save_model(model, DEFAULT_MODEL_PATH)
    print(f"モデルを {DEFAULT_MODEL_PATH} に保存しました。")


if __name__ == "__main__":
    ids = _read_ids(sys.argv[1:])
    if not ids:
        print("使い方: python train.py <race_id> [<race_id> ...] | --file ids.txt")
        sys.exit(1)
    main(ids)
