from keiba.models import Race, Horse, PastRun
from keiba.features import build_features

def _horse(name, odds, finishes):
    runs = [PastRun(date="2026-05-01", finish=f, course="東京", distance=1600,
                    surface="芝", track_condition="良", time=95.0, last_3f=33.5,
                    popularity=1, weight_carried=55.0, jockey="J", race_class="G3")
            for f in finishes]
    return Horse(name=name, sex="牡", age=4, weight_carried=56.0, jockey="J",
                 post=1, number=1, win_odds=odds, popularity=1, body_weight=480,
                 body_weight_diff=0, running_style=None, sire=None, dam=None,
                 broodmare_sire=None, training_time=None, training_course=None,
                 training_eval=None, past_runs=runs)

def test_build_features_columns_and_rows():
    race = Race(race_id="r1", name="t", date="d", course="東京", distance=1600,
                surface="芝", turn="右", track_condition="良", weather="晴",
                horses=[_horse("A", 2.0, [1, 2, 1]), _horse("B", 10.0, [5, 8, 6])])
    df = build_features(race)
    assert len(df) == 2
    assert "win_odds" in df.columns
    assert "avg_finish" in df.columns
    a = df[df["name"] == "A"].iloc[0]
    b = df[df["name"] == "B"].iloc[0]
    assert a["avg_finish"] < b["avg_finish"]


def test_build_features_race_condition_columns():
    race = Race(race_id="r1", name="t", date="d", course="東京", distance=1600,
                surface="芝", turn="右", track_condition="良", weather="晴",
                horses=[_horse("A", 2.0, [1, 2, 1])])
    df = build_features(race)
    row = df.iloc[0]
    assert row["distance"] == 1600
    assert row["surface_turf"] == 1
    assert row["track_condition_score"] == 0


def test_build_features_form_columns_from_past_runs():
    from keiba.form_features import FORM_COLUMNS
    race = Race(race_id="r1", name="t", date="d", course="東京", distance=1600,
                surface="芝", turn="右", track_condition="良", weather="晴",
                horses=[_horse("A", 2.0, [1, 2, 1])])
    row = build_features(race).iloc[0]
    for c in FORM_COLUMNS:
        assert c in row
    assert row["prev_runs"] == 3
    assert abs(row["prev_win_rate"] - 2 / 3) < 1e-9   # finishes 1,2,1 -> 2 wins
    assert abs(row["prev_avg_finish"] - 4 / 3) < 1e-9
    assert row["prev_avg_last3f"] == 33.5


def test_build_features_form_log_odds_and_first_timer():
    race = Race(race_id="r1", name="t", date="d", course="東京", distance=1600,
                surface="芝", turn="右", track_condition="良", weather="晴",
                horses=[_horse("A", 2.0, [1, 2]), _horse("B", 9.0, [])])
    # give A's past runs known odds -> mean log odds
    import math
    for r in race.horses[0].past_runs:
        r.win_odds = 4.0
    df = build_features(race)
    a = df[df["name"] == "A"].iloc[0]
    b = df[df["name"] == "B"].iloc[0]
    assert abs(a["prev_avg_log_odds"] - math.log(4.0)) < 1e-9
    # first-time runner -> neutral zeros
    assert b["prev_runs"] == 0
    assert b["prev_avg_log_odds"] == 0.0
    assert b["prev_win_rate"] == 0.0
