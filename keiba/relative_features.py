"""Derive within-race relative features from absolute ones (no network).

Racing signals are relative: an 8.0 win-odds horse is a favorite in a weak
field and an outsider in a strong one. These features normalize each horse
against the other runners in its own race.
"""
import numpy as np
import pandas as pd

# Columns this module adds; appended to the model's feature set.
RELATIVE_COLUMNS = ["log_odds", "odds_rank_pct", "body_weight_z",
                    "weight_carried_z", "field_size"]


def _zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def add_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with per-race relative feature columns added."""
    out = df.copy()
    out["log_odds"] = np.log(out["win_odds"].clip(lower=1.0))

    grp = out.groupby("race_id")
    # odds rank within race, scaled to [0, 1): 0 = strongest favorite.
    out["odds_rank_pct"] = grp["win_odds"].rank(method="first") - 1
    out["field_size"] = grp["win_odds"].transform("size")
    out["odds_rank_pct"] = out["odds_rank_pct"] / out["field_size"]

    out["body_weight_z"] = grp["body_weight"].transform(_zscore)
    out["weight_carried_z"] = grp["weight_carried"].transform(_zscore)
    return out
