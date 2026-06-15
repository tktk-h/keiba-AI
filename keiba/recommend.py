from keiba.expected_value import win_ev, ev
from keiba.predictor import (place_k, place_probabilities,
                             quinella_probabilities, wide_probabilities)
from keiba.confidence import confidence


def _runs_by_name(race):
    return {h.name: len(h.past_runs) for h in race.horses}


def _race_sharpness(win_probs):
    ps = sorted(win_probs.values(), reverse=True)
    if not ps or ps[0] <= 0:
        return 0.0
    second = ps[1] if len(ps) > 1 else 0.0
    return max(0.0, min(1.0, (ps[0] - second) / ps[0]))


def _field_ok(race):
    return 5 <= len(race.horses) <= 18


def predict_ranking(race, win_probs: dict) -> list:
    """全馬を推定勝率順に、複勝率・確信度つきで返す(①予想)。"""
    runs = _runs_by_name(race)
    sharp = _race_sharpness(win_probs)
    fok = _field_ok(race)
    k = place_k(len(race.horses))
    place = place_probabilities(win_probs, k) if k else {}
    odds_ok = {h.name: (h.win_odds is not None) for h in race.horses}
    rows = []
    for name, p in sorted(win_probs.items(), key=lambda kv: kv[1], reverse=True):
        score, level = confidence(runs.get(name, 0), sharp,
                                   odds_ok.get(name, False), fok)
        rows.append({"name": name, "win_prob": p,
                     "place_prob": place.get(name), "confidence": score,
                     "level": level})
    return rows


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


def _num2name(race):
    return {h.number: h.name for h in race.horses}


def recommend_all(race, win_probs: dict, odds: dict, top_n: int = 8):
    """単勝/複勝/ワイド/馬連を券種横断でEV順に並べ、(bets, any_positive)を返す。

    +EV(ev>=0)が無くても、最良候補は必ず返す(全レースで予想を提示)。
    """
    runs = _runs_by_name(race)
    sharp = _race_sharpness(win_probs)
    fok = _field_ok(race)
    num2name = _num2name(race)
    k = place_k(len(race.horses))
    place = place_probabilities(win_probs, k) if k else {}
    quin = quinella_probabilities(win_probs)
    wide = wide_probabilities(win_probs)
    win_odds = {h.name: h.win_odds for h in race.horses}

    bets = []

    for name, p in win_probs.items():
        o = win_odds.get(name)
        if o:
            cs, lv = confidence(runs.get(name, 0), sharp, True, fok)
            bets.append({"type": "単勝", "sel": name, "prob": p, "odds": o,
                         "ev": ev(p, o), "confidence": cs, "level": lv})

    for num, lohi in odds.get("place", {}).items():
        name = num2name.get(num)
        p = place.get(name)
        if name is None or p is None:
            continue
        lo = lohi[0]
        cs, lv = confidence(runs.get(name, 0), sharp, True, fok)
        bets.append({"type": "複勝", "sel": name, "prob": p, "odds": lo,
                     "ev": ev(p, lo), "confidence": cs, "level": lv})

    for kind, probmap, oddsmap in (("馬連", quin, odds.get("quinella", {})),
                                   ("ワイド", wide, odds.get("wide", {}))):
        for (na, nb), o in oddsmap.items():
            a, b = num2name.get(na), num2name.get(nb)
            if a is None or b is None:
                continue
            p = probmap.get(tuple(sorted((a, b))))
            if p is None:
                continue
            dr = min(runs.get(a, 0), runs.get(b, 0))
            cs, lv = confidence(dr, sharp, True, fok)
            bets.append({"type": kind, "sel": f"{na}-{nb}", "prob": p, "odds": o,
                         "ev": ev(p, o), "confidence": cs, "level": lv})

    bets.sort(key=lambda b: (b["ev"], b["confidence"]), reverse=True)
    any_positive = any(b["ev"] >= 0 for b in bets)
    return bets[:top_n], any_positive
