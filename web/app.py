"""ローカル競馬予想Webアプリ。/ で今日のレース一覧、/race/<id> で予想表示。"""
from flask import Flask, render_template, request, abort
from keiba.race_list import fetch_race_cards, today_cards
from keiba.report import build_race_report

app = Flask(__name__)
_cache = {}


@app.route("/")
def index():
    date = request.args.get("date")
    try:
        cards = fetch_race_cards(date) if date else today_cards()
    except Exception:  # noqa: BLE001
        cards = []
    venues = {}
    for c in cards:
        venues.setdefault(c["venue"], []).append(c)
    for lst in venues.values():
        lst.sort(key=lambda c: c["number"])
    return render_template("index.html", venues=venues, date=date)


@app.route("/race/<race_id>")
def race(race_id):
    if race_id not in _cache:
        try:
            _cache[race_id] = build_race_report(race_id)
        except Exception:  # noqa: BLE001
            abort(502)
    return render_template("race.html", r=_cache[race_id])
