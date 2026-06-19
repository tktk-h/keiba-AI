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


def test_assign_marks_strength_and_value():
    from keiba.recommend import assign_marks
    preds = [{"name": "A", "win_prob": 0.40, "score": 0},
             {"name": "B", "win_prob": 0.25, "score": 0},
             {"name": "C", "win_prob": 0.15, "score": 0},
             {"name": "D", "win_prob": 0.10, "score": 0},
             {"name": "E", "win_prob": 0.05, "score": 0},
             {"name": "F", "win_prob": 0.04, "score": 4},   # 人気薄だが妙味=穴
             {"name": "G", "win_prob": 0.01, "score": 0}]
    assign_marks(preds)
    m = {p["name"]: p["mark"] for p in preds}
    # 強さ上位5頭
    assert m["A"] == "◎" and m["B"] == "○" and m["C"] == "▲"
    assert m["D"] == "△" and m["E"] == "注"
    # 6番手以下で評価が高い馬=穴。残りは無印。
    assert m["F"] == "穴"
    assert m["G"] is None
    assert all(p["mark_reason"] for p in preds if p["mark"])


def test_assign_marks_no_ana_when_no_value_longshot():
    from keiba.recommend import assign_marks
    # 全馬ほぼ市場一致(score 0) -> 穴は付けない(強い馬に誤付けしない)
    preds = [{"name": chr(65 + i), "win_prob": 0.5 - i * 0.05, "score": 0}
             for i in range(7)]
    assign_marks(preds)
    marks = [p["mark"] for p in preds if p["mark"]]
    assert "穴" not in marks
    assert marks.count("◎") == 1 and len(marks) == 5   # ◎○▲△注のみ


def test_assign_marks_small_field():
    from keiba.recommend import assign_marks
    preds = [{"name": "A", "win_prob": 0.6, "score": 0},
             {"name": "B", "win_prob": 0.4, "score": 0}]
    assign_marks(preds)                          # クラッシュせず付けられる分だけ
    assert {p["name"]: p["mark"] for p in preds} == {"A": "◎", "B": "○"}
