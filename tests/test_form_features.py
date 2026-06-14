import pandas as pd
from keiba.form_features import add_form_features, FORM_COLUMNS


def _df():
    # Same horse "A" runs 3 times (chronological by race_id); "B" runs once.
    return pd.DataFrame([
        {"race_id": "202301010101", "name": "A", "won": 1,
         "popularity": 1, "win_odds": 2.0},
        {"race_id": "202301010101", "name": "B", "won": 0,
         "popularity": 3, "win_odds": 5.0},
        {"race_id": "202302010101", "name": "A", "won": 0,
         "popularity": 2, "win_odds": 4.0},
        {"race_id": "202303010101", "name": "A", "won": 1,
         "popularity": 1, "win_odds": 3.0},
    ])


def test_adds_form_columns():
    out = add_form_features(_df())
    for col in FORM_COLUMNS:
        assert col in out.columns


def test_prev_runs_counts_only_earlier_races():
    out = add_form_features(_df()).sort_values(["name", "race_id"])
    a = out[out["name"] == "A"].reset_index(drop=True)
    # A's three runs in order: 0, 1, 2 prior runs
    assert list(a["prev_runs"]) == [0, 1, 2]
    b = out[out["name"] == "B"].reset_index(drop=True)
    assert list(b["prev_runs"]) == [0]


def test_prev_win_rate_excludes_current_race():
    out = add_form_features(_df()).sort_values(["name", "race_id"]).reset_index(drop=True)
    a = out[out["name"] == "A"].reset_index(drop=True)
    # Run1: no history -> 0 (filled). Run2: after 1 win -> 1.0.
    # Run3: after 1 win + 1 loss -> 0.5.
    assert a.loc[0, "prev_win_rate"] == 0.0
    assert a.loc[1, "prev_win_rate"] == 1.0
    assert a.loc[2, "prev_win_rate"] == 0.5


def test_no_lookahead_on_first_run():
    # First-ever run should never reflect the current race's own result.
    out = add_form_features(_df())
    first_a = out[(out["name"] == "A") &
                  (out["race_id"] == "202301010101")].iloc[0]
    assert first_a["prev_runs"] == 0
    assert first_a["prev_win_rate"] == 0.0
