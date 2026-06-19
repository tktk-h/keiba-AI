import pandas as pd
from keiba.recommend import recommend_bets, deviation_score

def test_recommend_sorted_by_ev():
    df = pd.DataFrame({"name": ["A", "B"], "win_odds": [3.0, 10.0]})
    win_probs = {"A": 0.5, "B": 0.1}
    recs = recommend_bets(df, win_probs, top_n=2)
    assert recs[0]["ev"] >= recs[1]["ev"]
    assert recs[0]["bet_type"] == "単勝"
    assert "name" in recs[0]


def test_deviation_score_even_is_zero():
    assert deviation_score(0.3, 0.3) == 0


def test_deviation_score_undervalued_is_positive():
    # モデルが市場の1.3倍 -> +1 (市場が過小評価)
    assert deviation_score(0.39, 0.30) == 1


def test_deviation_score_overvalued_is_negative():
    # モデルが市場の 1/1.3 -> -1 (市場が過大評価)
    assert deviation_score(0.30, 0.39) == -1


def test_deviation_score_clamped_to_five():
    assert deviation_score(0.99, 0.01) == 5
    assert deviation_score(0.01, 0.99) == -5


def test_deviation_score_zero_when_missing():
    assert deviation_score(0.0, 0.3) == 0
    assert deviation_score(0.3, 0.0) == 0
    assert deviation_score(None, 0.3) == 0
