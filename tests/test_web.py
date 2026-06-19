import web.app as webapp


def test_index_lists_races(monkeypatch):
    monkeypatch.setattr(webapp, "today_cards", lambda: [
        {"race_id": "202605030411", "venue": "東京", "number": 11, "name": ""}])
    client = webapp.app.test_client()
    resp = client.get("/")
    body = resp.get_data(as_text=True)
    assert resp.status_code == 200
    assert "東京" in body
    assert "202605030411" in body


def _fake_report():
    return {"meta": {"race_id": "r", "name": "テストS", "surface": "芝",
                     "distance": 1600, "field_size": 2},
            "predictions": [{"name": "A", "win_prob": 0.6, "market_prob": 0.5,
                             "place_prob": 0.8, "value": "妙味", "score": 1,
                             "level": "高", "confidence": 0.8,
                             "reasons": [["近走着順", "↑"]]}],
            "bets": [{"type": "単勝", "sel": "A", "prob": 0.6, "odds": 2.0,
                      "ev": 0.2, "level": "高"}],
            "any_positive": True}


def test_race_shell_does_not_compute(monkeypatch):
    def boom(rid):
        raise AssertionError("shell route must not run build_race_report")
    monkeypatch.setattr(webapp, "build_race_report", boom)
    client = webapp.app.test_client()
    resp = client.get("/race/r")
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "/race/r/data" in body       # JSが取りに行くデータURLが埋まっている


def test_race_data_returns_json_with_score(monkeypatch):
    webapp._cache.clear()
    monkeypatch.setattr(webapp, "build_race_report", lambda rid: _fake_report())
    client = webapp.app.test_client()
    resp = client.get("/race/r/data")
    assert resp.status_code == 200
    assert resp.is_json
    data = resp.get_json()
    assert data["meta"]["name"] == "テストS"
    assert data["predictions"][0]["score"] == 1


def test_race_data_is_cached(monkeypatch):
    webapp._cache.clear()
    calls = []
    monkeypatch.setattr(webapp, "build_race_report",
                        lambda rid: calls.append(rid) or _fake_report())
    client = webapp.app.test_client()
    client.get("/race/r/data")
    client.get("/race/r/data")
    assert len(calls) == 1              # 2回目はキャッシュから
