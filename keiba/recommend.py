import math
from keiba.expected_value import win_ev, ev
from keiba.predictor import (place_k, place_probabilities,
                             quinella_probabilities, wide_probabilities)
from keiba.confidence import confidence, prediction_confidence


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


def _agreement(model_p: float, market_p: float) -> float:
    """モデル勝率と市場インプライド勝率の一致度 0–1(3倍ズレで0)。"""
    if model_p <= 0 or market_p <= 0:
        return 0.0
    return max(0.0, 1.0 - abs(math.log(model_p / market_p)) / math.log(3.0))


def predict_ranking(race, win_probs: dict) -> list:
    """全馬を推定勝率順に、複勝率・確信度つきで返す(①予想)。

    確信度は馬ごとに、過去走データ量と『モデルと市場(オッズ)の一致度』で決まる。
    """
    runs = _runs_by_name(race)
    fok = _field_ok(race)
    k = place_k(len(race.horses))
    place = place_probabilities(win_probs, k) if k else {}
    inv = {h.name: 1.0 / h.win_odds for h in race.horses if h.win_odds}
    z = sum(inv.values())
    market = {n: v / z for n, v in inv.items()} if z else {}
    rows = []
    for name, p in sorted(win_probs.items(), key=lambda kv: kv[1], reverse=True):
        agree = _agreement(p, market.get(name, 0.0))
        score, level = prediction_confidence(runs.get(name, 0), agree, fok)
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


def recommend_all(race, win_probs: dict, odds: dict, top_n: int = 8,
                  min_prob: float = 0.02, min_confidence: float = 0.40):
    """単勝/複勝/ワイド/馬連を券種横断でEV順に並べ、(bets, any_positive)を返す。

    当選確率が min_prob 未満、または確信度が min_confidence 未満の買い目は
    推奨から除外する(極小確率の穴連系のような、EV推定が脆い買い目を出さない)。
    フィルタ後に候補が無ければ空リストを返す。
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

    def _add(bet_type, sel, p, o, data_runs):
        if p is None or o is None or p < min_prob:
            return
        cs, lv = confidence(data_runs, sharp, True, fok, hit_prob=p)
        if cs < min_confidence:
            return
        bets.append({"type": bet_type, "sel": sel, "prob": p, "odds": o,
                     "ev": ev(p, o), "confidence": cs, "level": lv})

    for name, p in win_probs.items():
        _add("単勝", name, p, win_odds.get(name), runs.get(name, 0))

    for num, lohi in odds.get("place", {}).items():
        name = num2name.get(num)
        if name is None:
            continue
        _add("複勝", name, place.get(name), lohi[0], runs.get(name, 0))

    for kind, probmap, oddsmap in (("馬連", quin, odds.get("quinella", {})),
                                   ("ワイド", wide, odds.get("wide", {}))):
        for (na, nb), o in oddsmap.items():
            a, b = num2name.get(na), num2name.get(nb)
            if a is None or b is None:
                continue
            _add(kind, f"{na}-{nb}", probmap.get(tuple(sorted((a, b)))), o,
                 min(runs.get(a, 0), runs.get(b, 0)))

    bets.sort(key=lambda b: (b["ev"], b["confidence"]), reverse=True)
    any_positive = any(b["ev"] >= 0 for b in bets)
    return bets[:top_n], any_positive
