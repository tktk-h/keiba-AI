import json
from dataclasses import asdict
from pathlib import Path
from keiba.models import Race, Horse, PastRun

def save_race(race: Race, path) -> None:
    Path(path).write_text(
        json.dumps(asdict(race), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

def load_race(path) -> Race:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    horses = []
    for h in data["horses"]:
        past_runs = [PastRun(**pr) for pr in h.pop("past_runs", [])]
        horses.append(Horse(past_runs=past_runs, **h))
    data["horses"] = horses
    return Race(**data)
