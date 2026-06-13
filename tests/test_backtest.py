import pandas as pd
from keiba.backtest import time_split, simulate_win_bets


def _df(rows):
    return pd.DataFrame(rows)


def test_time_split_orders_by_race_id():
    df = _df([
        {"race_id": "202401010101", "name": "a", "won": 1, "win_odds": 2.0},
        {"race_id": "202412310101", "name": "b", "won": 0, "win_odds": 3.0},
        {"race_id": "202406010101", "name": "c", "won": 0, "win_odds": 5.0},
    ])
    train, test = time_split(df, test_fraction=0.34)
    # newest race goes to test set
    assert "202412310101" in test["race_id"].values
    assert "202412310101" not in train["race_id"].values
    assert len(train) + len(test) == 3


def test_simulate_win_bets_payout():
    # 2 races. We bet on the predicted top horse of each.
    # Race1: bet on winner at odds 4.0 -> payout 4.0 (won)
    # Race2: bet on loser -> payout 0
    df = _df([
        {"race_id": "R1", "name": "w", "won": 1, "win_odds": 4.0},
        {"race_id": "R1", "name": "x", "won": 0, "win_odds": 2.0},
        {"race_id": "R2", "name": "y", "won": 0, "win_odds": 3.0},
        {"race_id": "R2", "name": "z", "won": 1, "win_odds": 6.0},
    ])
    # pick: race -> bet horse name
    picks = {"R1": "w", "R2": "y"}
    result = simulate_win_bets(df, picks, stake=100)
    # bet 2 races x 100 = 200 spent; returned 400 from R1 only
    assert result["bets"] == 2
    assert result["spent"] == 200
    assert result["returned"] == 400
    assert result["hits"] == 1
    assert abs(result["roi"] - 2.0) < 1e-9  # 400/200


def test_simulate_skips_unknown_picks():
    df = _df([{"race_id": "R1", "name": "w", "won": 1, "win_odds": 4.0}])
    result = simulate_win_bets(df, {"R1": "missing"}, stake=100)
    assert result["bets"] == 1
    assert result["returned"] == 0
    assert result["hits"] == 0
