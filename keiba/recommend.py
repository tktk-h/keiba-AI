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
