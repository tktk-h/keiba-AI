from pathlib import Path
from keiba.scraper import parse_race

def test_parse_race_extracts_horses():
    html = Path("tests/fixtures/race_sample.html").read_text(encoding="utf-8")
    race = parse_race(html, race_id="202605010101")
    assert race.race_id == "202605010101"
    assert len(race.horses) >= 1
    first = race.horses[0]
    assert first.name
    assert first.number >= 1
