from keiba.expected_value import win_ev

def recommend_bets(df, win_probs: dict, top_n: int = 5) -> list:
    """単勝の期待値を計算し、期待値の高い順に上位top_n件を返す。"""
    bets = []
    for _, row in df.iterrows():
        name = row["name"]
        odds = row.get("win_odds")
        prob = win_probs.get(name)
        if odds and prob is not None:
            bets.append({
                "bet_type": "単勝",
                "name": name,
                "odds": odds,
                "prob": prob,
                "ev": win_ev(prob, odds),
            })
    bets.sort(key=lambda b: b["ev"], reverse=True)
    return bets[:top_n]
