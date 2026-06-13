from pathlib import Path
from keiba.horse_page import parse_results, parse_pedigree

# Fixtures are trimmed real netkeiba pages for horse 2021461... (2021105560):
#  - horse_result_sample.html: table.db_h_race_results, header + 3 rows
#  - horse_ped_sample.html: full table.blood_table (5-generation pedigree)


def _results():
    html = Path("tests/fixtures/horse_result_sample.html").read_text(encoding="utf-8")
    return parse_results(html, limit=5)


def test_parse_results_returns_past_runs():
    runs = _results()
    assert len(runs) == 3
    first = runs[0]
    assert first.date == "2025/11/30"
    assert first.finish == 15
    assert first.course == "東京"
    assert first.surface == "芝"
    assert first.distance == 2400
    assert first.track_condition == "良"
    assert first.popularity == 8
    assert first.weight_carried == 58.0
    assert first.jockey == "池添謙一"


def test_parse_results_time_in_seconds():
    first = _results()[0]
    # "2:23.0" -> 143.0 seconds
    assert abs(first.time - 143.0) < 1e-6


def test_parse_results_respects_limit():
    html = Path("tests/fixtures/horse_result_sample.html").read_text(encoding="utf-8")
    runs = parse_results(html, limit=2)
    assert len(runs) == 2


def test_parse_pedigree():
    html = Path("tests/fixtures/horse_ped_sample.html").read_text(encoding="utf-8")
    sire, dam, bms = parse_pedigree(html)
    assert sire == "レイデオロ"
    assert dam == "シャンドランジュ"
    assert bms == "マンハッタンカフェ"
