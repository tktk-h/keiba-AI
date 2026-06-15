import json
from pathlib import Path
from keiba.odds_page import (parse_odds, _extract_win, _extract_place,
                             _extract_pairs)


def _sample():
    return json.loads(Path("tests/fixtures/odds_sample.json").read_text(encoding="utf-8"))


def test_parse_odds_shapes():
    odds = parse_odds(_sample())
    # 複勝: 馬番(int) -> (下限float, 上限float)
    assert odds["place"][1] == (1.5, 1.8)
    # 馬連/ワイド: 昇順(int,int)タプル -> float
    assert odds["quinella"][(1, 2)] == 5.0
    assert odds["wide"][(1, 2)] == 2.5
    # キーは昇順正規化
    assert all(a < b for (a, b) in odds["quinella"])


def test_extract_win_from_block1_skips_scratched():
    # type=1 の block "1" が単勝 [odds, "0.0", 人気]
    raw = {"data": {"odds": {"1": {
        "01": ["2.2", "0.0", "1"],
        "02": ["1,250.9", "0.0", "16"],
        "16": ["-2.0", "0.0", "9999"],
    }}}}
    out = _extract_win(raw)
    assert out["01"] == "2.2"
    assert out["02"] == "1250.9"
    assert "16" not in out
    # parse_odds で int キー / float に
    parsed = parse_odds({"win": out})
    assert parsed["win"][1] == 2.2


def test_extract_place_skips_scratched_and_strips_commas():
    # type=1 の実構造: data.odds["2"] が複勝 [下限, 上限, 人気]
    raw = {"data": {"odds": {"2": {
        "01": ["1.5", "1.8", "1"],
        "02": ["1,234.5", "2,000.0", "18"],
        "16": ["-2.0", "-2.0", "9999"],   # 取消馬
    }}}}
    out = _extract_place(raw)
    assert out["01"] == ["1.5", "1.8"]
    assert out["02"] == ["1234.5", "2000.0"]
    assert "16" not in out


def test_extract_pairs_parses_4digit_combo_and_commas():
    # type=4 馬連の実構造: data.odds["4"] が {"0102": [odds, "0.0", 人気]}
    raw = {"data": {"odds": {"4": {
        "0102": ["5.0", "0.0", "1"],
        "0203": ["2,612.8", "0.0", "55"],
    }}}}
    out = _extract_pairs(raw, "4")
    assert out["1_2"] == "5.0"
    assert out["2_3"] == "2612.8"


def test_extract_to_parse_roundtrip():
    raw1 = {"data": {"odds": {"2": {"01": ["1.5", "1.8", "1"]}}}}
    raw4 = {"data": {"odds": {"4": {"0102": ["5.0", "0.0", "1"]}}}}
    raw5 = {"data": {"odds": {"5": {"0102": ["2.5", "3.0", "1"]}}}}
    sample = {"place": _extract_place(raw1),
              "quinella": _extract_pairs(raw4, "4"),
              "wide": _extract_pairs(raw5, "5")}
    odds = parse_odds(sample)
    assert odds["place"][1] == (1.5, 1.8)
    assert odds["quinella"][(1, 2)] == 5.0
    assert odds["wide"][(1, 2)] == 2.5
