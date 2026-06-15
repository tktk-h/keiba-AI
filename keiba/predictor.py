from itertools import permutations

def place_k(field_size: int):
    """JRA複勝の着内頭数: 8頭以上=3, 5-7頭=2, 4頭以下=None(複勝なし)。"""
    if field_size >= 8:
        return 3
    if field_size >= 5:
        return 2
    return None

def win_probabilities(df, model=None) -> dict:
    """各馬の勝率を返す。

    model を渡すと学習済みモデルで予測する(v2)。渡さなければ
    オッズベースの暫定勝率(1/odds を正規化)にフォールバックする(v1)。
    """
    if model is not None:
        from keiba.model import model_win_probabilities
        return model_win_probabilities(model, df)
    raw = {}
    for _, row in df.iterrows():
        odds = row.get("win_odds")
        raw[row["name"]] = (1.0 / odds) if odds and odds > 0 else 0.0
    total = sum(raw.values())
    if total == 0:
        n = len(raw)
        return {k: 1.0 / n for k in raw}
    return {k: v / total for k, v in raw.items()}

def place_probabilities(win: dict, k: int = 2) -> dict:
    """ハーヴィル法: k着以内に入る確率を勝率から近似算出。"""
    names = list(win.keys())
    result = {n: 0.0 for n in names}
    for order in permutations(names, k):
        remaining = dict(win)
        p = 1.0
        for horse in order:
            denom = sum(remaining.values())
            if denom == 0:
                p = 0.0
                break
            p *= remaining[horse] / denom
            del remaining[horse]
        for horse in order:
            result[horse] += p
    return result
