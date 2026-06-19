import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keiba.explain import feature_contributions


def _model():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [4.0, 3.0, 2.0, 1.0]})
    y = [0, 0, 1, 1]
    m = Pipeline([("scaler", StandardScaler()),
                  ("clf", LogisticRegression(max_iter=1000))])
    m.fit(X, y)
    m.feature_columns_ = ["a", "b"]
    return m


def test_feature_contributions_matches_coef_times_z():
    m = _model()
    row = {"a": 3.0, "b": 2.0}
    c = feature_contributions(m, row)
    sc = m.named_steps["scaler"]
    cl = m.named_steps["clf"]
    for i, k in enumerate(["a", "b"]):
        z = (row[k] - sc.mean_[i]) / sc.scale_[i]
        assert abs(c[k] - cl.coef_[0][i] * z) < 1e-9


def test_feature_contributions_missing_is_zero():
    m = _model()
    c = feature_contributions(m, {"a": 3.0})   # b 欠損
    assert c["b"] == 0.0


def test_deviation_reasons_excludes_market_and_sorts_by_abs():
    from keiba.explain import deviation_reasons
    contribs = {"win_odds": 5.0, "age": -0.4,
                "body_weight": 0.2, "prev_avg_finish": -0.9}
    r = deviation_reasons(contribs, top_n=2)
    feats = [f for f, _ in r]
    assert "win_odds" not in feats          # 市場要因は除外
    assert r[0][0] == "prev_avg_finish"     # |-0.9| が最大
    assert len(r) == 2


def test_deviation_reasons_drops_zero():
    from keiba.explain import deviation_reasons
    assert deviation_reasons({"age": 0.0, "body_weight": 0.0}) == []
