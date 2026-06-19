import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keiba.models import Race, Horse, PastRun
from keiba.report import assemble_report
from keiba.features import build_features


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
    assert all("score" in p for p in rep["predictions"])   # 乖離スコア
    assert all(-5 <= p["score"] <= 5 for p in rep["predictions"])


def test_meta_has_odds_flag():
    horses = [_horse("A", 1, 2.0), _horse("B", 2, 5.0), _horse("C", 3, 8.0),
              _horse("D", 4, 12.0), _horse("E", 5, 20.0)]
    race = Race(race_id="r1", name="t", date="d", course="東京", distance=1600,
                surface="芝", turn="右", track_condition="良", weather="晴",
                horses=horses)
    win = {h.name: 0.2 for h in horses}
    odds = {"win": {}, "place": {}, "quinella": {}, "wide": {}}
    assert assemble_report(race, win, odds)["meta"]["has_odds"] is True
    for h in race.horses:
        h.win_odds = None
    assert assemble_report(race, win, odds)["meta"]["has_odds"] is False


def _two_feature_model():
    X = pd.DataFrame({"age": [3.0, 4, 5, 6], "body_weight": [460.0, 480, 500, 520]})
    y = [0, 0, 1, 1]
    m = Pipeline([("scaler", StandardScaler()),
                  ("clf", LogisticRegression(max_iter=1000))])
    m.fit(X, y)
    m.feature_columns_ = ["age", "body_weight"]
    return m


def test_assemble_report_attaches_reasons_for_deviating():
    horses = [_horse("A", 1, 2.0), _horse("B", 2, 5.0), _horse("C", 3, 8.0),
              _horse("D", 4, 12.0), _horse("E", 5, 20.0)]
    race = Race(race_id="r1", name="テストS", date="d", course="東京",
                distance=1600, surface="芝", turn="右", track_condition="良",
                weather="晴", horses=horses)
    win = {h.name: p for h, p in zip(horses, [0.4, 0.25, 0.15, 0.12, 0.08])}
    odds = {"win": {}, "place": {}, "quinella": {}, "wide": {}}
    feats = build_features(race)
    model = _two_feature_model()
    rep = assemble_report(race, win, odds, model=model, features=feats)
    deviating = [p for p in rep["predictions"] if abs(p["score"]) >= 1]
    assert deviating, "テストデータに乖離馬が居ること"
    assert any("reasons" in p for p in deviating)
