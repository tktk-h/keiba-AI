from types import SimpleNamespace
from keiba.speed_figure import (build_baselines, condition_offsets,
                                speed_figure, horse_speed_rating)

BASE = {("芝", 2000): (120.0, 2.0)}  # mean 120s, std 2s


def test_speed_figure_at_baseline_mean_equals_base():
    assert speed_figure(120.0, "芝", 2000, BASE) == 50.0


def test_speed_figure_faster_time_gives_higher_figure():
    # 2s faster than the 120s mean (1 std) -> base + scale = 60.
    assert speed_figure(118.0, "芝", 2000, BASE) == 60.0
    # slower than mean -> below base.
    assert speed_figure(122.0, "芝", 2000, BASE) == 40.0


def test_speed_figure_none_when_no_baseline_or_missing_time():
    assert speed_figure(120.0, "ダ", 2000, BASE) is None
    assert speed_figure(None, "芝", 2000, BASE) is None


def test_build_baselines_groups_by_surface_distance():
    recs = [{"surface": "芝", "distance": 2000, "time": 120.0},
            {"surface": "芝", "distance": 2000, "time": 122.0}]
    out = build_baselines(recs, min_samples=2)
    mean, std = out[("芝", 2000)]
    assert mean == 121.0
    assert std == 1.0  # population std of [120, 122]


def test_build_baselines_skips_small_buckets():
    recs = [{"surface": "芝", "distance": 2000, "time": 120.0},
            {"surface": "芝", "distance": 2000, "time": 122.0},
            {"surface": "芝", "distance": 1600, "time": 95.0}]
    out = build_baselines(recs, min_samples=2)
    assert ("芝", 2000) in out
    assert ("芝", 1600) not in out  # only 1 sample


def test_condition_offsets_good_is_zero_and_slow_is_positive():
    recs = [{"surface": "芝", "distance": 2000, "time": 120.0, "track_condition": "良"},
            {"surface": "芝", "distance": 2000, "time": 122.0, "track_condition": "良"},
            {"surface": "芝", "distance": 2000, "time": 124.0, "track_condition": "重"},
            {"surface": "芝", "distance": 2000, "time": 126.0, "track_condition": "重"}]
    off = condition_offsets(recs)
    assert off["良"] == 0.0
    assert off["重"] == 4.0  # 重 mean 125 - 良 mean 121


def test_speed_figure_condition_offset_compensates_slow_ground():
    base = {("芝", 2000): (121.0, 2.0)}
    raw = speed_figure(125.0, "芝", 2000, base)
    adj = speed_figure(125.0, "芝", 2000, base,
                       track_condition="重", offsets={"重": 4.0})
    assert adj > raw          # slow-ground penalty removed
    assert adj == 50.0        # 125 - 4 = 121 == baseline mean


def test_horse_speed_rating_averages_recent_runs():
    runs = [{"time": 118.0, "surface": "芝", "distance": 2000, "track_condition": "良"},
            {"time": 122.0, "surface": "芝", "distance": 2000, "track_condition": "良"},
            {"time": 120.0, "surface": "芝", "distance": 2000, "track_condition": "良"}]
    # n=2 -> mean(figure(118)=60, figure(122)=40) = 50.
    assert horse_speed_rating(runs, BASE, n=2) == 50.0


def test_horse_speed_rating_skips_runs_without_baseline():
    runs = [{"time": 118.0, "surface": "ダ", "distance": 1200, "track_condition": "良"},
            {"time": 122.0, "surface": "芝", "distance": 2000, "track_condition": "良"}]
    # first run has no baseline -> only the 芝2000 run counts -> figure(122)=40.
    assert horse_speed_rating(runs, BASE, n=2) == 40.0


def test_horse_speed_rating_none_when_no_figures():
    runs = [{"time": 118.0, "surface": "ダ", "distance": 1200, "track_condition": "良"}]
    assert horse_speed_rating(runs, BASE, n=2) is None


def test_horse_speed_rating_accepts_pastrun_objects():
    runs = [SimpleNamespace(time=118.0, surface="芝", distance=2000,
                            track_condition="良")]
    assert horse_speed_rating(runs, BASE, n=2) == 60.0
