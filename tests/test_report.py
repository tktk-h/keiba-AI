from keiba.models import Race, Horse, PastRun
from keiba.report import assemble_report


def _horse(name, num, odds):
    runs = [PastRun(date="2026-01-01", finish=1, course="東京", distance=1600,
                    surface="芝", track_condition="良", time=95.0, last_3f=33.5,
                    popularity=1, weight_carried=55.0, jockey="J",
                    race_class="G3", win_odds=3.0) for _ in range(3)]
    return Horse(name=name, sex="牡", age=4, weight_carried=55.0, jockey="J",
                 post=num, number=num, win_odds=odds, popularity=num,
                 body_weight=480, body_weight_diff=0, running_style=None,
                 sire=None, dam=None, broodmare_sire=None, training_time=None,
                 training_course=None, training_eval=None, past_runs=runs)


def test_assemble_report_structure():
    horses = [_horse("A", 1, 2.0), _horse("B", 2, 5.0), _horse("C", 3, 8.0),
              _horse("D", 4, 12.0), _horse("E", 5, 20.0)]
    race = Race(race_id="r1", name="テストS", date="d", course="東京",
                distance=1600, surface="芝", turn="右", track_condition="良",
                weather="晴", horses=horses)
    win = {h.name: p for h, p in zip(horses, [0.4, 0.25, 0.15, 0.12, 0.08])}
    odds = {"win": {}, "place": {1: (1.5, 1.8)},
            "quinella": {(1, 2): 5.0}, "wide": {(1, 2): 2.5}}
    rep = assemble_report(race, win, odds)
    assert set(rep) == {"meta", "predictions", "bets", "any_positive"}
    assert rep["meta"]["race_id"] == "r1"
    assert rep["meta"]["name"] == "テストS"
    assert rep["meta"]["field_size"] == 5
    wp = [p["win_prob"] for p in rep["predictions"]]
    assert wp == sorted(wp, reverse=True)          # 勝率降順
    assert "value" in rep["predictions"][0]         # 妙味タグ
    evs = [b["ev"] for b in rep["bets"]]
    assert evs == sorted(evs, reverse=True)         # EV降順
