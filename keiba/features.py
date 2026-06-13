import pandas as pd
from keiba.models import Race

def _avg(values):
    nums = [v for v in values if v is not None]
    return sum(nums) / len(nums) if nums else None

def build_features(race: Race) -> pd.DataFrame:
    rows = []
    for h in race.horses:
        finishes = [r.finish for r in h.past_runs]
        last3f = [r.last_3f for r in h.past_runs]
        rows.append({
            "name": h.name,
            "number": h.number,
            "win_odds": h.win_odds,
            "age": h.age,
            "weight_carried": h.weight_carried,
            "avg_finish": _avg(finishes),
            "best_finish": min(finishes) if finishes else None,
            "avg_last_3f": _avg(last3f),
            "n_past_runs": len(h.past_runs),
        })
    return pd.DataFrame(rows)
