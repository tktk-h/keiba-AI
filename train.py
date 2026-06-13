"""Collect past race results from netkeiba and train the win model.

Usage:
    python train.py <race_id> [<race_id> ...]
    python train.py --file race_ids.txt   # one race_id per line

Saves the trained model to model.pkl. Run main.py afterwards to use it.
The more races you collect, the better the model — aim for hundreds.
"""
import sys
from keiba.dataset import collect_dataset
from keiba.model import train_model, save_model, DEFAULT_MODEL_PATH


def _read_ids(args):
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
