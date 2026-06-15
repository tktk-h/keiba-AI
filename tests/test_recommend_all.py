from keiba.models import Race, Horse, PastRun
from keiba.recommend import predict_ranking


def _horse(name, num, odds, n_runs):
    runs = [PastRun(date="2026-01-01", finish=1, course="東京", distance=1600,
                    surface="芝", track_condition="良", time=95.0, last_3f=33.5,
                    popularity=1, weight_carried=55.0, jockey="J",
                    race_class="G3", win_odds=3.0) for _ in range(n_runs)]
    return Horse(name=name, sex="牡", age=4, weight_carried=55.0, jockey="J",
                 post=num, number=num, win_odds=odds, popularity=num,
                 body_weight=480, body_weight_diff=0, running_style=None,
                 sire=None, dam=None, broodmare_sire=None, training_time=None,
                 training_course=None, training_eval=None, past_runs=runs)


def _race(n=8):
    horses = [_horse(chr(65 + i), i + 1, 2.0 + i, 3) for i in range(n)]
    return Race(race_id="r1", name="t", date="d", course="東京", distance=1600,
                surface="芝", turn="右", track_condition="良", weather="晴",
                horses=horses)


def test_predict_ranking_sorted_with_fields():
    race = _race(8)
    win = {h.name: (8 - i) / 36.0 for i, h in enumerate(race.horses)}
    rows = predict_ranking(race, win)
    assert len(rows) == 8
    probs = [r["win_prob"] for r in rows]
    assert probs == sorted(probs, reverse=True)
    top = rows[0]
    assert top["name"] == "A"
    assert 0.0 <= top["place_prob"] <= 1.0
    assert top["place_prob"] >= top["win_prob"]
    assert top["level"] in ("低", "中", "高")


from keiba.recommend import recommend_all


def test_recommend_all_ranks_by_ev_and_flags_positive():
    race = _race(8)
    win = {h.name: (8 - i) / 36.0 for i, h in enumerate(race.horses)}
    race.horses[0].win_odds = 99.0
    odds = {
        "place": {1: (1.5, 1.8)},
        "quinella": {(1, 2): 5.0},
        "wide": {(1, 2): 2.5},
    }
    bets, any_positive = recommend_all(race, win, odds, top_n=10)
    assert any_positive is True
    evs = [b["ev"] for b in bets]
    assert evs == sorted(evs, reverse=True)
    assert {b["type"] for b in bets} == {"単勝", "複勝", "馬連", "ワイド"}
    for b in bets:
        assert b["level"] in ("低", "中", "高")


def test_recommend_all_negative_when_low_odds():
    race = _race(8)
    win = {h.name: (8 - i) / 36.0 for i, h in enumerate(race.horses)}
    for h in race.horses:
        h.win_odds = 1.1
    odds = {"place": {}, "quinella": {}, "wide": {}}
    bets, any_positive = recommend_all(race, win, odds, top_n=5)
    assert any_positive is False
    assert len(bets) >= 1


def test_recommend_all_filters_microscopic_longshot_pairs():
    race = _race(8)
    probs = [0.7, 0.15, 0.06, 0.04, 0.02, 0.015, 0.01, 0.005]
    win = {h.name: probs[i] for i, h in enumerate(race.horses)}
    # 7-8番(超低確率)に高オッズ=計算上は+EVだが当選確率は0.01%程度
    odds = {"place": {}, "quinella": {(7, 8): 800.0}, "wide": {}}
    bets, _ = recommend_all(race, win, odds, top_n=20)
    assert "7-8" not in [b["sel"] for b in bets]      # フィルタで除外
    # しきい値を外せば候補には挙がる(=除外したのはフィルタだと確認)
    bets_raw, _ = recommend_all(race, win, odds, top_n=20,
                                min_prob=0.0, min_confidence=0.0)
    assert "7-8" in [b["sel"] for b in bets_raw]
