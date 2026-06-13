import pandas as pd
from keiba.relative_features import add_relative_features


def _df():
    return pd.DataFrame([
        {"race_id": "R1", "name": "a", "win_odds": 2.0,
         "body_weight": 500, "weight_carried": 57.0},
        {"race_id": "R1", "name": "b", "win_odds": 10.0,
         "body_weight": 480, "weight_carried": 55.0},
        {"race_id": "R1", "name": "c", "win_odds": 50.0,
         "body_weight": 460, "weight_carried": 54.0},
        {"race_id": "R2", "name": "d", "win_odds": 3.0,
         "body_weight": 470, "weight_carried": 56.0},
        {"race_id": "R2", "name": "e", "win_odds": 6.0,
         "body_weight": 470, "weight_carried": 56.0},
    ])


def test_adds_expected_columns():
    out = add_relative_features(_df())
    for col in ["log_odds", "odds_rank_pct", "body_weight_z",
                "weight_carried_z", "field_size"]:
        assert col in out.columns


def test_field_size_per_race():
    out = add_relative_features(_df())
    r1 = out[out["race_id"] == "R1"]
    r2 = out[out["race_id"] == "R2"]
    assert (r1["field_size"] == 3).all()
    assert (r2["field_size"] == 2).all()


def test_odds_rank_pct_favorite_lowest():
    out = add_relative_features(_df())
    r1 = out[out["race_id"] == "R1"].set_index("name")
    # favorite (lowest odds) has the smallest odds_rank_pct
    assert r1.loc["a", "odds_rank_pct"] < r1.loc["b", "odds_rank_pct"]
    assert r1.loc["b", "odds_rank_pct"] < r1.loc["c", "odds_rank_pct"]


def test_body_weight_z_zero_when_equal():
    out = add_relative_features(_df())
    r2 = out[out["race_id"] == "R2"]
    # both horses share the same body weight -> z-score 0
    assert (r2["body_weight_z"].abs() < 1e-9).all()


def test_log_odds_monotonic():
    out = add_relative_features(_df())
    r1 = out[out["race_id"] == "R1"].set_index("name")
    assert r1.loc["a", "log_odds"] < r1.loc["c", "log_odds"]
