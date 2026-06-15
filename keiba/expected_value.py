def ev(prob: float, odds: float) -> float:
    """期待値 = 確率 × オッズ − 1(EV≥0 は回収率100%以上)。"""
    return prob * odds - 1.0

def win_ev(prob: float, odds: float) -> float:
    """単勝期待値: 確率×オッズ - 1。"""
    return prob * odds - 1.0

def place_ev(prob: float, odds: float) -> float:
    """複勝期待値(複勝オッズを使用)。"""
    return prob * odds - 1.0

def combo_prob(probs: list) -> float:
    """組み合わせ確率の簡易計算(独立性を仮定して積を取る)。"""
    result = 1.0
    for p in probs:
        result *= p
    return result
