"""Shared encoders for race-condition fields (surface, track condition).

Used by both the training pipeline (keiba.dataset, built from
result_page rows) and the live-prediction pipeline (keiba.features, built
from a scraped Race) so the model sees the same encoding either way.
"""

_TRACK_CONDITION_SCALE = {"良": 0, "稍重": 1, "重": 2, "不良": 3}


def encode_surface(surface: str) -> int:
    """1 if turf (芝), 0 otherwise (ダ/障)."""
    return 1 if surface == "芝" else 0


def encode_track_condition(condition: str):
    """良=0 .. 不良=3 (worse footing = higher). Unknown -> None."""
    return _TRACK_CONDITION_SCALE.get(condition)
