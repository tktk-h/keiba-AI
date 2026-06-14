"""Build a training dataset (features + label) from race result rows.

`build_dataset` turns the list-of-dicts from result_page.parse_result_page
into a pandas DataFrame ready for model training. `collect_dataset` is the
network-bound helper that scrapes many races (manual use only).
"""
import time
import pandas as pd
from keiba.relative_features import add_relative_features, RELATIVE_COLUMNS
from keiba.form_features import add_form_features, FORM_COLUMNS

# Absolute per-horse signals (PRE-race only; last_3f / finish would leak).
BASE_FEATURE_COLUMNS = ["win_odds", "popularity", "age", "weight_carried",
                        "body_weight"]
# Full feature set the model trains on = absolute + within-race relative.
FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + RELATIVE_COLUMNS
LABEL_COLUMN = "won"


def build_dataset(rows) -> pd.DataFrame:
    """rows: flat list of dicts from parse_result_page. Returns a DataFrame
    with the feature columns + label, dropping rows missing any feature.

    Also includes FORM_COLUMNS (past-performance aggregates), computed from
    this row's own finish/last_3f for *earlier* races of the same horse.
    finish/last_3f themselves are dropped from the output (current-race
    values would leak)."""
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.dropna(subset=BASE_FEATURE_COLUMNS + [LABEL_COLUMN]).reset_index(drop=True)
    df = add_relative_features(df)
    df = add_form_features(df)
    keep = FEATURE_COLUMNS + FORM_COLUMNS + [LABEL_COLUMN, "race_id", "name"]
    return df[[c for c in keep if c in df.columns]].reset_index(drop=True)


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
