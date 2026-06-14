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


def train_model(df, feature_columns=FEATURE_COLUMNS) -> Pipeline:
    """Fit a logistic-regression pipeline on a dataset DataFrame.

    The chosen feature_columns are stored on the model so prediction uses the
    exact same set it was trained on.
    """
    X = df[feature_columns]
    y = df[LABEL_COLUMN]
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    model.fit(X, y)
    model.feature_columns_ = list(feature_columns)
    return model


def train_lgbm_model(df, feature_columns=FEATURE_COLUMNS):
    """Fit a LightGBM gradient-boosting classifier on a dataset DataFrame.

    Captures non-linear feature interactions the logistic model can't. Same
    interface as train_model: stores feature_columns_ and exposes predict_proba.
    """
    from lightgbm import LGBMClassifier
    X = df[feature_columns]
    y = df[LABEL_COLUMN]
    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=0,
        verbose=-1,
    )
    model.fit(X, y)
    model.feature_columns_ = list(feature_columns)
    return model


def save_model(model, path=DEFAULT_MODEL_PATH) -> None:
    Path(path).write_bytes(pickle.dumps(model))


def load_model(path=DEFAULT_MODEL_PATH):
    return pickle.loads(Path(path).read_bytes())


def model_win_probabilities(model, df) -> dict:
    """Predict normalized win probabilities for one race's feature DataFrame.

    Uses the model's own feature_columns_ (set at training). Rows missing any
    feature get a raw score of 0. Result sums to 1 (uniform if all zero).
    """
    feature_columns = getattr(model, "feature_columns_", FEATURE_COLUMNS)
    scores = {}
    for _, row in df.iterrows():
        feats = [row.get(c) for c in feature_columns]
        if any(v is None or (isinstance(v, float) and v != v) for v in feats):
            scores[row["name"]] = 0.0
            continue
        X = pd.DataFrame([feats], columns=feature_columns)
        prob_win = model.predict_proba(X)[0][1]
        scores[row["name"]] = float(prob_win)
    total = sum(scores.values())
    if total == 0:
        n = len(scores)
        return {k: 1.0 / n for k in scores} if n else {}
    return {k: v / total for k, v in scores.items()}
