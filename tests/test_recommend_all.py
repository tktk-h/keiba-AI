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
