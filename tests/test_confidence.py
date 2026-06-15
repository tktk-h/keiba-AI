from keiba.confidence import confidence


def test_confidence_high_when_rich_data_and_sharp():
    score, level = confidence(data_runs=5, sharpness=1.0,
                              odds_confirmed=True, field_ok=True)
    assert abs(score - 1.0) < 1e-9
    assert level == "高"


def test_confidence_low_when_no_data_and_muddled():
    score, level = confidence(data_runs=0, sharpness=0.0,
                              odds_confirmed=False, field_ok=False)
    assert score < 0.4
    assert level == "低"


def test_confidence_data_runs_capped_at_5():
    a, _ = confidence(data_runs=5, sharpness=0.5, odds_confirmed=True, field_ok=True)
    b, _ = confidence(data_runs=99, sharpness=0.5, odds_confirmed=True, field_ok=True)
    assert a == b
