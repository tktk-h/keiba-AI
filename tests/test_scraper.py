from pathlib import Path
from keiba.scraper import parse_race

# Fixture is a trimmed copy of a real netkeiba shutuba page
# (race_id 202405021211 = 2024 日本ダービー), keeping the real
# table.Shutuba_Table / tr.HorseList structure with the first 2 horses.


def _race():
    html = Path("tests/fixtures/race_sample.html").read_text(encoding="utf-8")
    return parse_race(html, race_id="202405021211")


def test_parse_race_extracts_horses():
    race = _race()
    assert race.race_id == "202405021211"
    assert len(race.horses) == 2
    first = race.horses[0]
    assert first.name == "サンライズアース"
    assert first.number == 1
    assert first.post == 1


def test_parse_race_extracts_horse_details():
    first = _race().horses[0]
    assert first.sex == "牡"
    assert first.age == 3
    assert first.weight_carried == 57.0
    assert first.jockey == "池添"
    assert first.body_weight == 524
    assert first.body_weight_diff == -8


def test_parse_race_extracts_meta():
    race = _race()
    assert race.name == "日本ダービー"
    assert race.surface == "芝"
    assert race.distance == 2400
    assert race.turn == "左"
    assert race.weather == "晴"
    assert race.track_condition == "良"


def test_odds_unavailable_parses_to_none():
    # Post-race shutuba page shows '---.-' / '**' for odds & popularity.
    first = _race().horses[0]
    assert first.win_odds is None
    assert first.popularity is None
