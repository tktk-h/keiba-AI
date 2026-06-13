import pandas as pd
from keiba.predictor import win_probabilities, place_probabilities

def test_win_probabilities_sum_to_one():
    df = pd.DataFrame({"name": ["A", "B", "C"], "win_odds": [2.0, 4.0, 8.0]})
    probs = win_probabilities(df)
    assert abs(sum(probs.values()) - 1.0) < 1e-6
    assert probs["A"] > probs["B"] > probs["C"]

def test_place_probabilities_harville():
    win = {"A": 0.5, "B": 0.3, "C": 0.2}
    place = place_probabilities(win, k=2)
    assert place["A"] > place["B"] > place["C"]
    assert all(0 <= p <= 1 for p in place.values())
