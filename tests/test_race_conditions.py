from keiba.race_conditions import encode_surface, encode_track_condition


def test_encode_surface_turf():
    assert encode_surface("芝") == 1


def test_encode_surface_dirt_and_jump():
    assert encode_surface("ダ") == 0
    assert encode_surface("障") == 0


def test_encode_track_condition_scale():
    assert encode_track_condition("良") == 0
    assert encode_track_condition("稍重") == 1
    assert encode_track_condition("重") == 2
    assert encode_track_condition("不良") == 3


def test_encode_track_condition_unknown_is_none():
    assert encode_track_condition("") is None
    assert encode_track_condition("謎") is None
