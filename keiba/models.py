from dataclasses import dataclass, field
from typing import Optional

@dataclass
class PastRun:
    date: str
    finish: int
    course: str
    distance: int
    surface: str
    track_condition: str
    time: Optional[float]
    last_3f: Optional[float]
    popularity: Optional[int]
    weight_carried: Optional[float]
    jockey: Optional[str]
    race_class: Optional[str]
    win_odds: Optional[float] = None

@dataclass
class Horse:
    name: str
    sex: str
    age: int
    weight_carried: float
    jockey: str
    post: int
    number: int
    win_odds: Optional[float]
    popularity: Optional[int]
    body_weight: Optional[int]
    body_weight_diff: Optional[int]
    running_style: Optional[str]
    sire: Optional[str]
    dam: Optional[str]
    broodmare_sire: Optional[str]
    training_time: Optional[str]
    training_course: Optional[str]
    training_eval: Optional[str]
    horse_id: Optional[str] = None
    past_runs: list = field(default_factory=list)

@dataclass
class Race:
    race_id: str
    name: str
    date: str
    course: str
    distance: int
    surface: str
    turn: str
    track_condition: str
    weather: str
    horses: list = field(default_factory=list)
