from itertools import permutations

def win_probabilities(df) -> dict:
    """オッズベースの暫定勝率(1/odds を正規化)。"""
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
