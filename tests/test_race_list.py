from pathlib import Path
from keiba.race_list import parse_kaisai_dates, parse_race_ids


def test_parse_kaisai_dates():
    html = Path("tests/fixtures/calendar_sample.html").read_text(encoding="utf-8")
    dates = parse_kaisai_dates(html)
    assert "20240526" in dates
    assert len(dates) == 8
    # sorted, no duplicates
    assert dates == sorted(set(dates))
    assert all(len(d) == 8 and d.isdigit() for d in dates)


def test_parse_race_ids():
    html = Path("tests/fixtures/race_list_sample.html").read_text(encoding="utf-8")
    ids = parse_race_ids(html)
    assert len(ids) == 24
    assert "202405021201" in ids
    assert ids == sorted(set(ids))
    assert all(len(i) == 12 and i.isdigit() for i in ids)
