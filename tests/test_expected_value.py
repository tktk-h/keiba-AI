from keiba.expected_value import win_ev, place_ev

def test_win_ev_positive_when_underpriced():
    assert abs(win_ev(prob=0.5, odds=3.0) - 0.5) < 1e-9

def test_win_ev_negative_when_overpriced():
    assert win_ev(prob=0.2, odds=3.0) < 0

def test_place_ev():
    assert abs(place_ev(prob=0.6, odds=2.0) - 0.2) < 1e-9
