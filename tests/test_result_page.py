from pathlib import Path
from keiba.result_page import parse_result_page

# Fixture: trimmed real netkeiba race result page (race 202405021211,
# 2024 日本ダービー) — table.race_table_01, header + 3 finishers.


def _rows():
    html = Path("tests/fixtures/race_result_page_sample.html").read_text(encoding="utf-8")
    return parse_result_page(html, race_id="202405021211")


def test_parse_result_page_rows():
    rows = _rows()
    assert len(rows) == 3
    winner = rows[0]
    assert winner["race_id"] == "202405021211"
    assert winner["name"] == "ダノンデサイル"
    assert winner["horse_id"] == "2021105143"
    assert winner["finish"] == 1
    assert winner["won"] == 1


def test_parse_result_page_features():
    winner = _rows()[0]
    assert winner["win_odds"] == 46.6
    assert winner["popularity"] == 9
    assert winner["age"] == 3
    assert winner["sex"] == "牡"
    assert winner["weight_carried"] == 57.0
    assert winner["last_3f"] == 33.5
    assert winner["body_weight"] == 504


def test_won_flag_only_for_first():
    rows = _rows()
    assert rows[0]["won"] == 1
    assert all(r["won"] == 0 for r in rows[1:])
