"""Collect race datasets month-by-month, saving each month's CSV as it
finishes (so an interrupted run can resume from the last completed month).

Usage:
    python collect_months.py <start YYYY-MM> <end YYYY-MM> [--suffix v2]

Saves data_months/<YYYY_MM><suffix>.csv for each month. Skips months whose
output file already exists.
"""
import sys
from pathlib import Path
from keiba.race_list import collect_race_ids
from keiba.dataset import collect_dataset


def _months_in_range(start, end):
    sy, sm = (int(x) for x in start.split("-"))
    ey, em = (int(x) for x in end.split("-"))
    y, m = sy, sm
    while (y, m) <= (ey, em):
        yield y, m
        m += 1
        if m > 12:
            y, m = y + 1, 1


def main(start, end, suffix):
    out_dir = Path("data_months")
    out_dir.mkdir(exist_ok=True)
    for year, month in _months_in_range(start, end):
        out_path = out_dir / f"{year}_{month:02d}{suffix}.csv"
        if out_path.exists():
            print(f"  {year}-{month:02d}: 既存({out_path}) -> スキップ")
            continue
        print(f"  {year}-{month:02d} のレースID収集中...")
        ids = collect_race_ids(year, month)
        print(f"  {len(ids)} レースの結果を収集中(時間がかかります)...")
        df = collect_dataset(ids)
        df.to_csv(out_path, index=False)
        print(f"  -> {out_path} に保存({len(df)}行)")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) < 2:
        print(__doc__)
        sys.exit(1)
    suffix = "_v2"
    if "--suffix" in sys.argv:
        suffix = "_" + sys.argv[sys.argv.index("--suffix") + 1]
    main(args[0], args[1], suffix)
