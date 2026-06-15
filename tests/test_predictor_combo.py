from keiba.predictor import place_k


def test_place_k_rules():
    assert place_k(8) == 3
    assert place_k(18) == 3
    assert place_k(7) == 2
    assert place_k(5) == 2
    assert place_k(4) is None
    assert place_k(1) is None
