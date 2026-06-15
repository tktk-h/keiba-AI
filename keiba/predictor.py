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

def _candidate_names(win: dict, top_n: int):
    """確率上位 top_n 頭の名前(組み合わせ計算を現実的な規模に抑える)。"""
    ordered = sorted(win.items(), key=lambda kv: kv[1], reverse=True)
    return [name for name, _ in ordered[:top_n]]


def quinella_probabilities(win: dict, top_n: int = 12) -> dict:
    """馬連: 2頭がともに上位2着に入る確率(順不同)。キーは昇順タプル。

    ハーヴィル法。条件付き分母は全馬の勝率和 W を使う(上位 top_n のみ
    列挙するが、末尾の馬を含むペアは確率がほぼ0なので無視できる)。
    """
    names = _candidate_names(win, top_n)
    total = sum(win.values())
    out = {}
    for a, b in permutations(names, 2):
        d1 = total - win[a]
        if total <= 0 or d1 <= 0:   # degenerate distribution -> 0 contribution
            continue
        p = (win[a] / total) * (win[b] / d1)
        key = tuple(sorted((a, b)))
        out[key] = out.get(key, 0.0) + p
    return out


def wide_probabilities(win: dict, top_n: int = 12) -> dict:
    """ワイド: 2頭がともに上位3着に入る確率(順不同)。キーは昇順タプル。

    上位3着の順列を列挙し、その集合に含まれる各ペアへ確率を加算する。
    """
    names = _candidate_names(win, top_n)
    total = sum(win.values())
    out = {}
    for a, b, c in permutations(names, 3):
        d1 = total - win[a]
        d2 = total - win[a] - win[b]
        if total <= 0 or d1 <= 0 or d2 <= 0:   # degenerate -> 0 contribution
            continue
        p = (win[a] / total) * (win[b] / d1) * (win[c] / d2)
        for x, y in ((a, b), (a, c), (b, c)):
            key = tuple(sorted((x, y)))
            out[key] = out.get(key, 0.0) + p
    return out
