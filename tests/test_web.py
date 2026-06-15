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


def test_race_page_renders(monkeypatch):
    fake = {"meta": {"race_id": "r", "name": "テストS", "surface": "芝",
                     "distance": 1600, "field_size": 2},
            "predictions": [{"name": "A", "win_prob": 0.6, "market_prob": 0.5,
                             "place_prob": 0.8, "value": "妙味", "level": "高",
                             "confidence": 0.8}],
            "bets": [{"type": "単勝", "sel": "A", "prob": 0.6, "odds": 2.0,
                      "ev": 0.2, "level": "高"}],
            "any_positive": True}
    webapp._cache.clear()
    monkeypatch.setattr(webapp, "build_race_report", lambda rid: fake)
    client = webapp.app.test_client()
    resp = client.get("/race/r")
    body = resp.get_data(as_text=True)
    assert resp.status_code == 200
    assert "テストS" in body
    assert "妙味" in body
