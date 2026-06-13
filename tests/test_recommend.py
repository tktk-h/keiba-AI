import pandas as pd
from keiba.recommend import recommend_bets

def test_recommend_sorted_by_ev():
    df = pd.DataFrame({"name": ["A", "B"], "win_odds": [3.0, 10.0]})
    win_probs = {"A": 0.5, "B": 0.1}
    recs = recommend_bets(df, win_probs, top_n=2)
    assert recs[0]["ev"] >= recs[1]["ev"]
    assert recs[0]["bet_type"] == "単勝"
    assert "name" in recs[0]
