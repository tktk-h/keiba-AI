"""Train / save / load a win-probability model and predict with it.

v2 model: logistic regression over numeric features (FEATURE_COLUMNS),
labeled by whether the horse won. Predicted per-horse win scores are
normalized within a race so they sum to 1 — a drop-in replacement for the
odds-based estimate in predictor.win_probabilities.
"""
import pickle
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keiba.dataset import FEATURE_COLUMNS, LABEL_COLUMN

DEFAULT_MODEL_PATH = "model.pkl"


def train_model(df) -> Pipeline:
    """Fit a logistic-regression pipeline on a dataset DataFrame."""
    X = df[FEATURE_COLUMNS]
    y = df[LABEL_COLUMN]
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    model.fit(X, y)
    return model


def save_model(model, path=DEFAULT_MODEL_PATH) -> None:
    Path(path).write_bytes(pickle.dumps(model))


def load_model(path=DEFAULT_MODEL_PATH):
    return pickle.loads(Path(path).read_bytes())


def model_win_probabilities(model, df) -> dict:
    """Predict normalized win probabilities for one race's feature DataFrame.

    df must contain FEATURE_COLUMNS and a 'name' column. Rows missing any
    feature get a raw score of 0. Result sums to 1 (uniform if all zero).
    """
    scores = {}
    for _, row in df.iterrows():
        feats = [row.get(c) for c in FEATURE_COLUMNS]
        if any(v is None or (isinstance(v, float) and v != v) for v in feats):
            scores[row["name"]] = 0.0
            continue
        X = pd.DataFrame([feats], columns=FEATURE_COLUMNS)
        prob_win = model.predict_proba(X)[0][1]
        scores[row["name"]] = float(prob_win)
    total = sum(scores.values())
    if total == 0:
        n = len(scores)
        return {k: 1.0 / n for k in scores} if n else {}
    return {k: v / total for k, v in scores.items()}
