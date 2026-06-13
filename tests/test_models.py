from keiba.models import PastRun, Horse, Race

def test_horse_holds_basic_and_past_runs():
    past = PastRun(date="2026-05-01", finish=1, course="東京", distance=1600,
                   surface="芝", track_condition="良", time=95.2, last_3f=33.5,
                   popularity=2, weight_carried=55.0, jockey="ルメール", race_class="G3")
    horse = Horse(name="テスト馬", sex="牡", age=4, weight_carried=56.0,
                  jockey="ルメール", post=3, number=5, win_odds=4.2, popularity=2,
                  body_weight=480, body_weight_diff=2, running_style="先行",
                  sire="ディープインパクト", dam="テスト母", broodmare_sire="母父馬",
                  training_time=None, training_course=None, training_eval=None,
                  past_runs=[past])
    assert horse.past_runs[0].finish == 1
    assert horse.win_odds == 4.2

def test_race_holds_horses():
    race = Race(race_id="202605010101", name="テストS", date="2026-05-01",
                course="東京", distance=1600, surface="芝", turn="右",
                track_condition="良", weather="晴", horses=[])
    assert race.race_id == "202605010101"
    assert race.horses == []
