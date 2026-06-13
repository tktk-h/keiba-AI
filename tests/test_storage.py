from keiba.models import Race, Horse, PastRun
from keiba.storage import save_race, load_race

def _sample_race():
    past = PastRun(date="2026-05-01", finish=1, course="東京", distance=1600,
                   surface="芝", track_condition="良", time=95.2, last_3f=33.5,
                   popularity=2, weight_carried=55.0, jockey="ルメール", race_class="G3")
    horse = Horse(name="テスト馬", sex="牡", age=4, weight_carried=56.0,
                  jockey="ルメール", post=3, number=5, win_odds=4.2, popularity=2,
                  body_weight=480, body_weight_diff=2, running_style="先行",
                  sire="ディープ", dam="母", broodmare_sire="母父",
                  training_time=None, training_course=None, training_eval=None,
                  past_runs=[past])
    return Race(race_id="202605010101", name="テストS", date="2026-05-01",
                course="東京", distance=1600, surface="芝", turn="右",
                track_condition="良", weather="晴", horses=[horse])

def test_save_and_load_roundtrip(tmp_path):
    race = _sample_race()
    path = tmp_path / "race.json"
    save_race(race, path)
    loaded = load_race(path)
    assert loaded.race_id == race.race_id
    assert loaded.horses[0].name == "テスト馬"
    assert loaded.horses[0].past_runs[0].finish == 1
