import math
import pandas as pd
from keiba.models import Race
from keiba.relative_features import add_relative_features
from keiba.race_conditions import encode_surface, encode_track_condition
from keiba.form_features import FORM_COLUMNS

def _avg(values):
    nums = [v for v in values if v is not None]
    return sum(nums) / len(nums) if nums else None


def _form_from_past_runs(past_runs, last3f_fill):
    """Aggregate a horse's prior runs into FORM_COLUMNS, mirroring the
    training-side add_form_features (means over prior races; first-timers
    get prev_runs=0 and neutral 0.0 rates)."""
    n = len(past_runs)
    if n == 0:
        return {c: 0.0 for c in FORM_COLUMNS}  # prev_runs becomes 0.0
    wins = [1 if r.finish == 1 else 0 for r in past_runs]
    logodds = [math.log(max(r.win_odds, 1.0))
               for r in past_runs if r.win_odds is not None]
    l3 = [(r.last_3f if r.last_3f is not None else last3f_fill) for r in past_runs]
    return {
        "prev_runs": n,
        "prev_win_rate": sum(wins) / n,
        "prev_avg_popularity": _avg([r.popularity for r in past_runs]) or 0.0,
        "prev_avg_log_odds": sum(logodds) / len(logodds) if logodds else 0.0,
        "prev_avg_finish": _avg([r.finish for r in past_runs]) or 0.0,
        "prev_avg_last3f": sum(l3) / len(l3),
    }

def build_features(race: Race) -> pd.DataFrame:
    # Field-wide mean of last_3f to fill missing values before averaging,
    # matching add_form_features (0 would skew toward "very fast").
    all_l3 = [r.last_3f for h in race.horses for r in h.past_runs
              if r.last_3f is not None]
    last3f_fill = sum(all_l3) / len(all_l3) if all_l3 else 0.0
    rows = []
    for h in race.horses:
        finishes = [r.finish for r in h.past_runs if r.finish is not None]
        last3f = [r.last_3f for r in h.past_runs]
        rows.append({
            "race_id": race.race_id,
            "name": h.name,
            "number": h.number,
            "win_odds": h.win_odds,
            "popularity": h.popularity,
            "body_weight": h.body_weight,
            "age": h.age,
            "weight_carried": h.weight_carried,
            "avg_finish": _avg(finishes),
            "best_finish": min(finishes) if finishes else None,
            "avg_last_3f": _avg(last3f),
            "n_past_runs": len(h.past_runs),
            "distance": race.distance,
            "surface_turf": encode_surface(race.surface),
            "track_condition_score": encode_track_condition(race.track_condition),
            **_form_from_past_runs(h.past_runs, last3f_fill),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        # 馬体重は発走直前、馬場も当日まで未確定なことがある。前売りオッズが出て
        # いれば予想を出せるよう、欠損を中立値で補完する(影響は極小)。オッズ自体が
        # 無い場合は win_odds が欠損のままなので、勝率は一様(=オッズ待ち)になる。
        if df["body_weight"].notna().any():
            df["body_weight"] = df["body_weight"].fillna(df["body_weight"].mean())
        else:
            df["body_weight"] = df["body_weight"].fillna(480.0)
        df["track_condition_score"] = df["track_condition_score"].fillna(0)
    # Add within-race relative features when the inputs are present, so the
    # trained model receives the same columns it learned on.
    if not df.empty and df["win_odds"].notna().any():
        df = add_relative_features(df)
    return df
