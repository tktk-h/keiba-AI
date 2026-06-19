"""ローカル競馬予想Webアプリ。/ で今日のレース一覧、/race/<id> で予想表示。

/race/<id> は即座にシェル(スピナー)を返し、重い予想計算は /race/<id>/data
(JSON)で行う。スマホでブラウザがブランクのまま固まらないようにするため。
"""
from datetime import date as _date
from flask import Flask, render_template, request, abort, jsonify
from keiba.race_list import fetch_race_cards, today_cards, fetch_kaisai_dates
from keiba.report import build_race_report

app = Flask(__name__)
_cache = {}


@app.after_request
def _no_cache(resp):
    # スマホが古いHTML/JSをキャッシュして更新が反映されない問題を防ぐ。
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp


def _norm_date(raw):
    """'2026-06-21' や '2026/06/21' を 'YYYYMMDD' に正規化。8桁でなければ None。"""
    if not raw:
        return None
    digits = "".join(c for c in raw if c.isdigit())
    return digits if len(digits) == 8 else None


def _upcoming_kaisai(n=4):
    """今日以降の開催日(YYYYMMDD)を最大 n 件。開催の無い日に次の開催へ誘導する。"""
    today = _date.today()
    nxt = (today.year + today.month // 12, today.month % 12 + 1)
    dates = []
    for y, mo in [(today.year, today.month), nxt]:
        try:
            dates += fetch_kaisai_dates(y, mo)
        except Exception:  # noqa: BLE001
            pass
    ts = today.strftime("%Y%m%d")
    fut = sorted(d for d in set(dates) if d >= ts)[:n]
    return [{"d": d, "label": f"{int(d[4:6])}/{int(d[6:])}"} for d in fut]


@app.route("/")
def index():
    date = _norm_date(request.args.get("date"))
    try:
        cards = fetch_race_cards(date) if date else today_cards()
    except Exception:  # noqa: BLE001
        cards = []
    venues = {}
    for c in cards:
        venues.setdefault(c["venue"], []).append(c)
    for lst in venues.values():
        lst.sort(key=lambda c: c["number"])
    upcoming = _upcoming_kaisai() if not venues else []
    date_iso = f"{date[:4]}-{date[4:6]}-{date[6:]}" if date else ""
    return render_template("index.html", venues=venues, date=date,
                           date_iso=date_iso, upcoming=upcoming)


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
