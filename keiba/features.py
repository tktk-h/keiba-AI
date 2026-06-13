import pandas as pd
from keiba.models import Race
from keiba.relative_features import add_relative_features

def _avg(values):
    nums = [v for v in values if v is not None]
    return sum(nums) / len(nums) if nums else None

def build_features(race: Race) -> pd.DataFrame:
    rows = []
    for h in race.horses:
        finishes = [r.finish for r in h.past_runs]
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
        })
    df = pd.DataFrame(rows)
    # Add within-race relative features when the inputs are present, so the
    # trained model receives the same columns it learned on.
    if not df.empty and df["win_odds"].notna().any():
        df = add_relative_features(df)
    return df
