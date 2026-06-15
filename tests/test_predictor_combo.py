from keiba.predictor import place_k
from keiba.predictor import quinella_probabilities
from keiba.predictor import wide_probabilities


def test_place_k_rules():
    assert place_k(8) == 3
    assert place_k(18) == 3
    assert place_k(7) == 2
    assert place_k(5) == 2
    assert place_k(4) is None
    assert place_k(1) is None


def test_quinella_probabilities_basic():
    win = {"A": 0.5, "B": 0.3, "C": 0.2}
    q = quinella_probabilities(win)
    assert set(q.keys()) == {("A", "B"), ("A", "C"), ("B", "C")}
    assert abs(sum(q.values()) - 1.0) < 1e-9
    expected_ab = (0.5 * 0.3) / (1 - 0.5) + (0.3 * 0.5) / (1 - 0.3)
    assert abs(q[("A", "B")] - expected_ab) < 1e-9
    assert q[("A", "B")] == max(q.values())


def test_wide_ge_quinella():
    win = {"A": 0.4, "B": 0.3, "C": 0.2, "D": 0.1}
    q = quinella_probabilities(win)
    w = wide_probabilities(win)
    for pair in q:
        assert w[pair] >= q[pair] - 1e-12
    for v in w.values():
        assert 0.0 <= v <= 1.0 + 1e-12
