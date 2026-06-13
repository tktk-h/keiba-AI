import pandas as pd
from keiba.dataset import build_dataset
from keiba.model import (train_model, save_model, load_model,
                         model_win_probabilities)


def _rows():
    # Synthetic: lower odds / popularity number => more likely to win.
    rows = []
    for race in range(40):
        # favorite wins
        rows.append(dict(race_id=f"r{race}", name=f"fav{race}", finish=1, won=1,
                         sex="牡", age=4, win_odds=2.0, popularity=1,
                         weight_carried=55.0, last_3f=33.0, body_weight=480))
        rows.append(dict(race_id=f"r{race}", name=f"mid{race}", finish=2, won=0,
                         sex="牡", age=4, win_odds=8.0, popularity=5,
                         weight_carried=55.0, last_3f=34.5, body_weight=470))
        rows.append(dict(race_id=f"r{race}", name=f"out{race}", finish=3, won=0,
                         sex="牡", age=4, win_odds=30.0, popularity=12,
                         weight_carried=55.0, last_3f=36.0, body_weight=460))
    return rows


def test_build_dataset_shape():
    df = build_dataset(_rows())
    assert len(df) == 120
    assert "won" in df.columns
    assert "win_odds" in df.columns


def test_train_and_predict_favorite_highest():
    df = build_dataset(_rows())
    model = train_model(df)
    race = pd.DataFrame([
        {"name": "A", "win_odds": 2.0, "popularity": 1, "age": 4,
         "weight_carried": 55.0, "last_3f": 33.0, "body_weight": 480},
        {"name": "B", "win_odds": 30.0, "popularity": 12, "age": 4,
         "weight_carried": 55.0, "last_3f": 36.0, "body_weight": 460},
    ])
    probs = model_win_probabilities(model, race)
    assert abs(sum(probs.values()) - 1.0) < 1e-6
    assert probs["A"] > probs["B"]


def test_save_and_load_model(tmp_path):
    df = build_dataset(_rows())
    model = train_model(df)
    path = tmp_path / "m.pkl"
    save_model(model, path)
    loaded = load_model(path)
    race = pd.DataFrame([
        {"name": "A", "win_odds": 2.0, "popularity": 1, "age": 4,
         "weight_carried": 55.0, "last_3f": 33.0, "body_weight": 480},
    ])
    from keiba.dataset import FEATURE_COLUMNS
    X = pd.DataFrame([[2.0, 1, 4, 55.0, 480]], columns=FEATURE_COLUMNS)
    probs = loaded.predict_proba(X)
    assert 0 <= probs[0][1] <= 1
    assert model_win_probabilities(loaded, race)["A"] == 1.0
