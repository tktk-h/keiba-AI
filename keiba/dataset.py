"""Build a training dataset (features + label) from race result rows.

`build_dataset` turns the list-of-dicts from result_page.parse_result_page
into a pandas DataFrame ready for model training. `collect_dataset` is the
network-bound helper that scrapes many races (manual use only).
"""
import time
import pandas as pd

# Numeric feature columns the model trains on. `won` is the label.
# Only PRE-race signals — last_3f / finish are race outcomes (would leak).
FEATURE_COLUMNS = ["win_odds", "popularity", "age", "weight_carried",
                   "body_weight"]
LABEL_COLUMN = "won"


def build_dataset(rows) -> pd.DataFrame:
    """rows: flat list of dicts from parse_result_page. Returns a DataFrame
    with the feature columns + label, dropping rows missing any feature."""
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    keep = FEATURE_COLUMNS + [LABEL_COLUMN, "race_id", "name"]
    df = df[[c for c in keep if c in df.columns]]
    return df.dropna(subset=FEATURE_COLUMNS + [LABEL_COLUMN]).reset_index(drop=True)


def collect_dataset(race_ids, delay: float = 1.0) -> pd.DataFrame:
    """Scrape many result pages into one training DataFrame (network)."""
    from keiba.result_page import fetch_result_rows
    all_rows = []
    for rid in race_ids:
        try:
            all_rows.extend(fetch_result_rows(rid))
        except Exception as exc:  # noqa: BLE001
            print(f"  ! レース {rid} の取得に失敗: {exc}")
        time.sleep(delay)
    return build_dataset(all_rows)
