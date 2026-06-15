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


def test_confidence_low_hit_prob_penalized():
    base, _ = confidence(5, 1.0, True, True)                 # 買い目確率を渡さない
    hi, _ = confidence(5, 1.0, True, True, hit_prob=0.2)     # 当たりやすい買い目
    lo, _ = confidence(5, 1.0, True, True, hit_prob=0.002)   # 極小確率の穴
    assert hi == base          # 5%以上は満点扱いで減点なし
    assert lo < hi             # 極小確率はEV推定が脆く、確信度が下がる
