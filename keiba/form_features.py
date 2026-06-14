"""Past-performance (form) features built from the dataset itself.

We already hold every race result over our window, so a horse's prior runs
are mostly already in the data. For each (race, horse) we aggregate ONLY that
horse's earlier races — no extra scraping, and no look-ahead leakage because
the current and future races are excluded via shift().

Chronology is approximated by sorting on race_id, which is monotonic with the
meeting calendar (YYYY + track + meeting + day + race).
"""
import numpy as np
import pandas as pd

FORM_COLUMNS = ["prev_runs", "prev_win_rate", "prev_avg_popularity",
                "prev_avg_log_odds", "prev_avg_finish", "prev_avg_last3f"]


def add_form_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with per-horse prior-form columns added.

    First-time runners get prev_runs=0 and neutral (0.0) rates.
    """
    out = df.copy()
    out["_order"] = np.arange(len(out))  # stable tiebreaker
    out = out.sort_values(["name", "race_id", "_order"]).reset_index(drop=True)
    out["_log_odds"] = np.log(out["win_odds"].clip(lower=1.0))
    # Fill missing last_3f with the field average before aggregating, since
    # 0 would skew a horse's average toward "very fast" instead of "unknown".
    out["_last3f"] = out["last_3f"].fillna(out["last_3f"].mean())

    grp = out.groupby("name", sort=False)
    # cumcount = number of strictly-earlier runs for this horse.
    out["prev_runs"] = grp.cumcount()

    # Expanding mean over prior rows only = (cumulative mean) shifted by one.
    def _prior_mean(col):
        cs = grp[col].cumsum() - out[col]
        return cs / out["prev_runs"].where(out["prev_runs"] > 0)

    out["prev_win_rate"] = _prior_mean("won")
    out["prev_avg_popularity"] = _prior_mean("popularity")
    out["prev_avg_log_odds"] = _prior_mean("_log_odds")
    out["prev_avg_finish"] = _prior_mean("finish")
    out["prev_avg_last3f"] = _prior_mean("_last3f")

    # First-time runners (prev_runs == 0) -> neutral 0.0.
    for col in ["prev_win_rate", "prev_avg_popularity", "prev_avg_log_odds",
                "prev_avg_finish", "prev_avg_last3f"]:
        out[col] = out[col].fillna(0.0)

    out = out.sort_values("_order").reset_index(drop=True)
    return out.drop(columns=["_order", "_log_odds", "_last3f"])
