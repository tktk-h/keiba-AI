"""Distance/surface/going-normalized speed figures from finishing times.

A speed figure expresses how fast a run was relative to a baseline time for
that (surface, distance), adjusted for track condition (going). Higher = faster.

Unlike raw form, it is a *quantitative* past-performance signal that the win
odds do not fully encode — the intended new feature for nudging predictions
away from a pure odds reproduction.

Baselines are data-driven (built from a corpus of past times). Track-condition
offsets normalize slow ground to a 良-equivalent basis so figures compare
across goings.
"""


def _get(run, key):
    """Read `key` from a dict or an object (PastRun-style)."""
    if isinstance(run, dict):
        return run.get(key)
    return getattr(run, key, None)


def _std(values, mean):
    """Population standard deviation (ddof=0), matching relative_features."""
    return (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5


def build_baselines(records, min_samples=5):
    """{(surface, distance): (mean_time, std_time)} from a corpus of runs.

    records: iterable of dicts/objects with surface, distance, time (seconds).
    Buckets with fewer than `min_samples` valid times are omitted; runs with a
    missing time are ignored.
    """
    buckets = {}
    for r in records:
        t = _get(r, "time")
        if t is None:
            continue
        key = (_get(r, "surface"), _get(r, "distance"))
        buckets.setdefault(key, []).append(float(t))
    out = {}
    for key, times in buckets.items():
        if len(times) < min_samples:
            continue
        mean = sum(times) / len(times)
        out[key] = (mean, _std(times, mean))
    return out


def condition_offsets(records):
    """{track_condition: offset_seconds} estimating how much slower each going
    is than 良, averaged over shared (surface, distance) buckets.

    Positive = slower ground. 良 is 0.0 by definition. A condition is only
    measured against 良 within the same bucket (so distance mix doesn't bias it).
    """
    # bucket -> condition -> [times]
    grid = {}
    for r in records:
        t = _get(r, "time")
        if t is None:
            continue
        key = (_get(r, "surface"), _get(r, "distance"))
        cond = _get(r, "track_condition")
        grid.setdefault(key, {}).setdefault(cond, []).append(float(t))

    diffs = {}  # condition -> [offset per bucket]
    for conds in grid.values():
        good = conds.get("良")
        if not good:
            continue
        good_mean = sum(good) / len(good)
        for cond, times in conds.items():
            if cond == "良":
                continue
            diffs.setdefault(cond, []).append(sum(times) / len(times) - good_mean)

    out = {"良": 0.0}
    for cond, vals in diffs.items():
        out[cond] = sum(vals) / len(vals)
    return out


def speed_figure(time, surface, distance, baselines, track_condition=None,
                 offsets=None, base=50.0, scale=10.0):
    """Standardized speed figure for one run (higher = faster), or None.

    figure = base + scale * (baseline_mean - adj_time) / baseline_std
    where adj_time removes the going penalty: adj_time = time - offset(going).
    Returns None when the time is missing, there is no baseline for the
    (surface, distance), or the baseline std is zero.
    """
    if time is None:
        return None
    stats = baselines.get((surface, distance))
    if stats is None:
        return None
    mean, std = stats
    if std == 0:
        return None
    adj = float(time)
    if offsets and track_condition in offsets:
        adj -= offsets[track_condition]
    return base + scale * (mean - adj) / std


def horse_speed_rating(past_runs, baselines, offsets=None, n=5):
    """Average speed figure over a horse's most-recent `n` past runs.

    past_runs: dicts/objects with time, surface, distance, track_condition
    (newest first). Returns None if no run yields a figure.
    """
    figures = []
    for run in past_runs[:n]:
        fig = speed_figure(_get(run, "time"), _get(run, "surface"),
                           _get(run, "distance"), baselines,
                           track_condition=_get(run, "track_condition"),
                           offsets=offsets)
        if fig is not None:
            figures.append(fig)
    if not figures:
        return None
    return sum(figures) / len(figures)
