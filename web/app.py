"""ローカル競馬予想Webアプリ。/ で今日のレース一覧、/race/<id> で予想表示。

/race/<id> は即座にシェル(スピナー)を返し、重い予想計算は /race/<id>/data
(JSON)で行う。スマホでブラウザがブランクのまま固まらないようにするため。
"""
from flask import Flask, render_template, request, abort, jsonify
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
    # 重い計算はせず、シェル(スピナー)だけ即返す。データは /data から取る。
    return render_template("race.html", race_id=race_id)


@app.route("/race/<race_id>/data")
def race_data(race_id):
    if race_id not in _cache:
        try:
            _cache[race_id] = build_race_report(race_id)
        except Exception:  # noqa: BLE001
            abort(502)
    return jsonify(_cache[race_id])
