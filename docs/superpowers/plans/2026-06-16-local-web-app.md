# ローカル競馬予想Webアプリ Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** ブラウザで今日の開催レースを一覧から選ぶだけで、①予想(モデル vs 市場・妙味)と②EV参考を表示するローカルWebアプリを作る。

**Architecture:** 既存のCLIパイプライン(`main.py`)を共通関数 `keiba/report.build_race_report` に切り出し、CLIとWeb(Flask)の両方がそれを呼ぶ。Flaskが今日のレース一覧と各レースの結果ページをサーバーサイドレンダリングし、結果はメモリキャッシュする。ローカル起動(localhost)で完結。

**Tech Stack:** Python, Flask + Jinja2(新規), 既存の requests/bs4/pandas/scikit-learn。

---

## File Structure

- `keiba/report.py`(新規): `assemble_report`(純粋・組み立て)と `build_race_report`(ネットワーク・全工程)。
- `main.py`(変更): `build_race_report` を呼ぶ薄い表示層に。
- `keiba/race_list.py`(変更): `race_card_from_id` / `parse_race_cards` / `fetch_race_cards` / `today_cards` を追加。
- `web/app.py`(新規): Flask アプリ(`/` 一覧、`/race/<race_id>` 結果)。
- `web/templates/index.html`, `web/templates/race.html`(新規)。
- `run_web.py`(新規): 起動スクリプト。
- `requirements.txt`(変更): `flask` を追加。
- テスト: `tests/test_report.py`, `tests/test_race_cards.py`, `tests/test_web.py`。

既存の `keiba/recommend.py`(`predict_ranking`, `recommend_all`)、`keiba/scraper.py`、`keiba/features.py`、`keiba/predictor.py`(`win_probabilities`)、`keiba/odds_page.py`(`fetch_odds`)、`keiba/model.py`(`load_model`, `DEFAULT_MODEL_PATH`)はそのまま利用する。

---

## Task 1: 共通レポート組み立て `keiba/report.py`

CLIとWebが共有する組み立て関数。`assemble_report` は純粋(テスト対象)、`build_race_report` はネットワーク全工程。

**Files:**
- Create: `keiba/report.py`
- Test: `tests/test_report.py`

- [ ] **Step 1: 失敗するテストを書く**

```python
# tests/test_report.py
from keiba.models import Race, Horse, PastRun
from keiba.report import assemble_report


def _horse(name, num, odds):
    runs = [PastRun(date="2026-01-01", finish=1, course="東京", distance=1600,
                    surface="芝", track_condition="良", time=95.0, last_3f=33.5,
                    popularity=1, weight_carried=55.0, jockey="J",
                    race_class="G3", win_odds=3.0) for _ in range(3)]
    return Horse(name=name, sex="牡", age=4, weight_carried=55.0, jockey="J",
                 post=num, number=num, win_odds=odds, popularity=num,
                 body_weight=480, body_weight_diff=0, running_style=None,
                 sire=None, dam=None, broodmare_sire=None, training_time=None,
                 training_course=None, training_eval=None, past_runs=runs)


def test_assemble_report_structure():
    horses = [_horse("A", 1, 2.0), _horse("B", 2, 5.0), _horse("C", 3, 8.0),
              _horse("D", 4, 12.0), _horse("E", 5, 20.0)]
    race = Race(race_id="r1", name="テストS", date="d", course="東京",
                distance=1600, surface="芝", turn="右", track_condition="良",
                weather="晴", horses=horses)
    win = {h.name: p for h, p in zip(horses, [0.4, 0.25, 0.15, 0.12, 0.08])}
    odds = {"win": {}, "place": {1: (1.5, 1.8)},
            "quinella": {(1, 2): 5.0}, "wide": {(1, 2): 2.5}}
    rep = assemble_report(race, win, odds)
    assert set(rep) == {"meta", "predictions", "bets", "any_positive"}
    assert rep["meta"]["race_id"] == "r1"
    assert rep["meta"]["name"] == "テストS"
    assert rep["meta"]["field_size"] == 5
    wp = [p["win_prob"] for p in rep["predictions"]]
    assert wp == sorted(wp, reverse=True)          # 勝率降順
    assert "value" in rep["predictions"][0]         # 妙味タグ
    evs = [b["ev"] for b in rep["bets"]]
    assert evs == sorted(evs, reverse=True)         # EV降順
```

- [ ] **Step 2: 失敗を確認**

Run: `PYTHONIOENCODING=utf-8 python -m pytest tests/test_report.py -v`
Expected: FAIL（`No module named 'keiba.report'`）

- [ ] **Step 3: 最小実装**

```python
# keiba/report.py
"""CLI と Web が共有する、1レース分のレポート組み立て。

assemble_report は純粋(テスト対象)。build_race_report はネットワーク全工程
(出馬表→確定オッズ補完→特徴量→モデル→オッズ→組み立て)。
"""
import os
from keiba.scraper import scrape_race
from keiba.features import build_features
from keiba.predictor import win_probabilities
from keiba.odds_page import fetch_odds
from keiba.model import load_model, DEFAULT_MODEL_PATH
from keiba.recommend import predict_ranking, recommend_all


def assemble_report(race, win_probs: dict, odds: dict) -> dict:
    """予想ランキングと買い目を、表示しやすい dict にまとめる。"""
    predictions = predict_ranking(race, win_probs)
    bets, any_positive = recommend_all(race, win_probs, odds)
    return {
        "meta": {
            "race_id": race.race_id,
            "name": race.name,
            "surface": race.surface,
            "distance": race.distance,
            "field_size": len(race.horses),
        },
        "predictions": predictions,
        "bets": bets,
        "any_positive": any_positive,
    }


def _backfill_market(race, odds) -> None:
    """締め切り後などで出馬表に単勝/人気が無い場合、確定オッズで補完する。"""
    for h in race.horses:
        if h.win_odds is None and odds["win"].get(h.number) is not None:
            h.win_odds = odds["win"][h.number]
    runners = [h for h in race.horses if h.win_odds is not None]
    if runners and all(h.popularity is None for h in race.horses):
        for rank, h in enumerate(sorted(runners, key=lambda x: x.win_odds), 1):
            h.popularity = rank


def build_race_report(race_id: str, enrich: bool = True) -> dict:
    """1レースを取得・予測して assemble_report の dict を返す(ネットワーク)。"""
    race = scrape_race(race_id, enrich=enrich)
    try:
        odds = fetch_odds(race_id)
    except Exception:  # noqa: BLE001
        odds = {"win": {}, "place": {}, "quinella": {}, "wide": {}}
    _backfill_market(race, odds)
    df = build_features(race)
    model = (load_model(DEFAULT_MODEL_PATH)
             if os.path.exists(DEFAULT_MODEL_PATH) else None)
    win_probs = win_probabilities(df, model=model)
    return assemble_report(race, win_probs, odds)
```

- [ ] **Step 4: テストパスを確認**

Run: `PYTHONIOENCODING=utf-8 python -m pytest tests/test_report.py -v`
Expected: PASS

- [ ] **Step 5: コミット**

```bash
git add keiba/report.py tests/test_report.py
git commit -m "feat: shared race-report builder (assemble_report + build_race_report)"
```

---

## Task 2: `main.py` を共通関数へ委譲

CLIの処理を `build_race_report` に委譲し、表示だけ行う薄い層にする(DRY)。

**Files:**
- Modify: `main.py`(全置換)

- [ ] **Step 1: main.py を全置換**

```python
import sys
from keiba.report import build_race_report


def main(race_id: str, enrich: bool = False):
    print(f"レース {race_id} を取得・計算中...")
    if enrich:
        print("  各馬の過去成績も取得します(時間がかかります)...")
    rep = build_race_report(race_id, enrich=enrich)
    m = rep["meta"]
    print(f"  {m['name']} / {m['surface']}{m['distance']}m / {m['field_size']}頭")

    print("\n=== ① 予想(モデル vs 市場・妙味つき) ===")
    print("  予想=モデル勝率 / 市場=オッズが示す勝率 / 妙味=モデルが人気より高評価")
    for i, r in enumerate(rep["predictions"], 1):
        pp = f"{r['place_prob']:.1%}" if r["place_prob"] is not None else "—"
        print(f"{i:>2}. {r['name']:<11} 予想{r['win_prob']:5.1%} 市場{r['market_prob']:5.1%} "
              f"{r['value']:<5}複勝{pp:>6} 確信{r['level']}")

    print("\n=== ② 買い目のEV(参考・優位性は未確認) ===")
    print("※ 250レースのバックテストでは控除率を越える優位性は確認されていません"
          "(回収率~75%)。EVは目安です。")
    if not rep["bets"]:
        print("推奨できる買い目はありません(確率・確信度が基準を満たす候補なし)。")
        return
    if not rep["any_positive"]:
        print("※ +EVの買い目はありません。今レースは見送り推奨(以下は参考)。")
    for i, b in enumerate(rep["bets"], 1):
        mark = "＋" if b["ev"] >= 0 else "－"
        print(f"{i}. [{b['type']}] {b['sel']:<11} オッズ{b['odds']:>6} "
              f"推定{b['prob']:5.1%} 期待値{b['ev']:+.2f}{mark} 確信{b['level']}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    enrich = "--enrich" in sys.argv
    if not args:
        print("使い方: python main.py <race_id> [--enrich]")
        sys.exit(1)
    main(args[0], enrich=enrich)
```

- [ ] **Step 2: import と全テストを確認**

Run: `PYTHONIOENCODING=utf-8 python -c "import main" && PYTHONIOENCODING=utf-8 python -m pytest -q`
Expected: import エラー無し、全テスト PASS

- [ ] **Step 3: コミット**

```bash
git add main.py
git commit -m "refactor: main CLI delegates to build_race_report (DRY)"
```

---

## Task 3: 今日のレース一覧 `race_list.fetch_race_cards`

**Files:**
- Modify: `keiba/race_list.py`
- Test: `tests/test_race_cards.py`

- [ ] **Step 1: 失敗するテストを書く**

```python
# tests/test_race_cards.py
from keiba.race_list import race_card_from_id, parse_race_cards


def test_race_card_from_id_derives_venue_and_number():
    c = race_card_from_id("202605030411")   # 05=東京, 末尾11=11R
    assert c["race_id"] == "202605030411"
    assert c["venue"] == "東京"
    assert c["number"] == 11
    assert c["name"] == ""                    # v1は名称未取得(空)


def test_parse_race_cards_from_list_html():
    html = ('<a href="/race/shutuba.html?race_id=202605030411">x</a>'
            '<a href="/race/shutuba.html?race_id=202609030401">y</a>')
    cards = parse_race_cards(html)
    ids = {c["race_id"] for c in cards}
    assert ids == {"202605030411", "202609030401"}
    venues = {c["venue"] for c in cards}
    assert venues == {"東京", "阪神"}           # 05=東京, 09=阪神
```

- [ ] **Step 2: 失敗を確認**

Run: `PYTHONIOENCODING=utf-8 python -m pytest tests/test_race_cards.py -v`
Expected: FAIL（`cannot import name 'race_card_from_id'`）

- [ ] **Step 3: 最小実装**

`keiba/race_list.py` の末尾に追加(既存の `parse_race_ids` を再利用):

```python
from datetime import date

VENUES = {"01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
          "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"}


def race_card_from_id(race_id: str) -> dict:
    """race_id から会場・R番号を導出。race_id=YYYY+場(2)+開催(2)+日(2)+R(2)。"""
    return {"race_id": race_id,
            "venue": VENUES.get(race_id[4:6], race_id[4:6]),
            "number": int(race_id[10:12]),
            "name": ""}


def parse_race_cards(html: str) -> list:
    """一覧ページHTMLから {race_id, venue, number, name} のリストを返す。"""
    return [race_card_from_id(rid) for rid in parse_race_ids(html)]


def fetch_race_cards(kaisai_date: str) -> list:
    """指定日(YYYYMMDD)の開催レースカード一覧(ネットワーク)。"""
    url = f"https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={kaisai_date}"
    return parse_race_cards(fetch_html(url))


def today_cards() -> list:
    """今日(ローカル日付)の開催レースカード一覧(ネットワーク)。"""
    return fetch_race_cards(date.today().strftime("%Y%m%d"))
```

- [ ] **Step 4: テストパスを確認**

Run: `PYTHONIOENCODING=utf-8 python -m pytest tests/test_race_cards.py -v`
Expected: PASS

- [ ] **Step 5: コミット**

```bash
git add keiba/race_list.py tests/test_race_cards.py
git commit -m "feat: race cards for a date (race_id -> venue/number)"
```

---

## Task 4: Flask アプリ + テンプレート

**Files:**
- Create: `web/app.py`, `web/templates/index.html`, `web/templates/race.html`
- Modify: `requirements.txt`
- Test: `tests/test_web.py`

- [ ] **Step 1: flask を入れる**

`requirements.txt` の末尾に1行 `flask` を追加し、インストール:

Run: `pip install flask`
Expected: 正常終了（既に入っていれば "already satisfied"）

- [ ] **Step 2: 失敗するテストを書く**

```python
# tests/test_web.py
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
```

- [ ] **Step 3: 失敗を確認**

Run: `PYTHONIOENCODING=utf-8 python -m pytest tests/test_web.py -v`
Expected: FAIL（`No module named 'web'` または `web.app`）

- [ ] **Step 4: Flask アプリを実装**

```python
# web/app.py
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
```

- [ ] **Step 5: テンプレートを作る**

```html
<!-- web/templates/index.html -->
<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>競馬予想</title>
<style>body{font-family:sans-serif;margin:1rem;max-width:720px}
a.card{display:block;padding:.6rem;border:1px solid #ccc;border-radius:6px;
margin:.25rem 0;text-decoration:none;color:#0366d6}
h2{margin-top:1.5rem}</style></head>
<body>
<h1>本日のレース</h1>
{% if not venues %}
<p>本日の開催はありません(URL に <code>?date=YYYYMMDD</code> で日付指定も可)。</p>
{% endif %}
{% for venue, cards in venues.items() %}
<h2>{{ venue }}</h2>
{% for c in cards %}
<a class="card" href="/race/{{ c.race_id }}">{{ c.number }}R{% if c.name %} {{ c.name }}{% endif %}</a>
{% endfor %}
{% endfor %}
</body></html>
```

```html
<!-- web/templates/race.html -->
<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ r.meta.name }}</title>
<style>body{font-family:sans-serif;margin:1rem;max-width:760px}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{border:1px solid #ddd;padding:4px 6px;text-align:right}
td.l,th.l{text-align:left}.tag{font-weight:bold}
.note{color:#888;font-size:12px}a{color:#0366d6}</style></head>
<body>
<p><a href="/">← レース一覧</a></p>
<h1>{{ r.meta.name }}
<small>{{ r.meta.surface }}{{ r.meta.distance }}m / {{ r.meta.field_size }}頭</small></h1>

<h2>① 予想(モデル vs 市場・妙味)</h2>
<table>
<tr><th class="l">馬</th><th>予想</th><th>市場</th><th>妙味</th><th>複勝</th><th>確信</th></tr>
{% for p in r.predictions %}
<tr><td class="l">{{ p.name }}</td>
<td>{{ '%.1f%%'|format(p.win_prob*100) }}</td>
<td>{{ '%.1f%%'|format(p.market_prob*100) }}</td>
<td class="tag">{{ p.value }}</td>
<td>{% if p.place_prob is not none %}{{ '%.1f%%'|format(p.place_prob*100) }}{% else %}—{% endif %}</td>
<td>{{ p.level }}</td></tr>
{% endfor %}
</table>

<h2>② 買い目のEV(参考・優位性は未確認)</h2>
<p class="note">250レースのバックテストで控除率を越える優位性は確認されていません
(回収率~75%)。EVは目安です。</p>
{% if not r.bets %}<p>推奨できる買い目はありません。</p>{% else %}
<table>
<tr><th class="l">券種</th><th class="l">買い目</th><th>オッズ</th><th>推定</th><th>EV</th><th>確信</th></tr>
{% for b in r.bets %}
<tr><td class="l">{{ b.type }}</td><td class="l">{{ b.sel }}</td>
<td>{{ b.odds }}</td>
<td>{{ '%.1f%%'|format(b.prob*100) }}</td>
<td>{{ '%+.2f'|format(b.ev) }}</td>
<td>{{ b.level }}</td></tr>
{% endfor %}
</table>{% endif %}
</body></html>
```

- [ ] **Step 6: テストパスを確認**

Run: `PYTHONIOENCODING=utf-8 python -m pytest tests/test_web.py -v`
Expected: PASS（2件）

- [ ] **Step 7: 全テストの非破壊を確認**

Run: `PYTHONIOENCODING=utf-8 python -m pytest -q`
Expected: 全 PASS

- [ ] **Step 8: コミット**

```bash
git add web/app.py web/templates/index.html web/templates/race.html requirements.txt tests/test_web.py
git commit -m "feat: Flask web app (today's races list + race report page)"
```

---

## Task 5: 起動スクリプトと手動確認

**Files:**
- Create: `run_web.py`

- [ ] **Step 1: 起動スクリプトを作る**

```python
# run_web.py
"""ローカルWebアプリを起動: http://localhost:5000 をブラウザで開く。"""
from web.app import app

if __name__ == "__main__":
    print("http://localhost:5000 をブラウザで開いてください(終了は Ctrl+C)")
    app.run(host="127.0.0.1", port=5000, debug=False)
```

- [ ] **Step 2: 構文確認**

Run: `PYTHONIOENCODING=utf-8 python -c "import run_web"`
Expected: エラー無し（サーバーは起動しない=`__main__` ではない）

- [ ] **Step 3: 手動スモーク(ネットワーク・対話)**

開催のある日付で起動して確認する(非開催日は `?date=YYYYMMDD` を使う):

Run: `PYTHONIOENCODING=utf-8 python run_web.py`
ブラウザで `http://localhost:5000/?date=20260614` を開く → 函館/東京/阪神のレースが一覧表示される。
1つクリック → 30〜60秒後に ①予想(妙味)と②EV参考が表示される。再度開くと即時(キャッシュ)。
確認後 Ctrl+C で停止。

- [ ] **Step 4: コミット**

```bash
git add run_web.py
git commit -m "feat: run_web.py launcher for the local web app"
```

---

## Self-Review(著者チェック済み)

- **仕様カバレッジ**: 共通レポート(report.py)=Task1、CLI委譲=Task2、
  今日のレース一覧(fetch_race_cards/today_cards)=Task3、Flask `/`・`/race/<id>`
  +テンプレート+キャッシュ=Task4、run_web.py起動=Task5。`?date=` 上書き・
  「本日の開催はありません」表示=Task4 index。スコープ外(スマホ公開・非同期・
  カレンダー・3連系)は未実装で正しい。
- **プレースホルダ**: 純粋部・テンプレートは完全コードを記載。Task5 Step3 のみ
  ネットワーク手動確認(性質上、対話必須)。
- **型整合**: `assemble_report -> {meta,predictions,bets,any_positive}` を Task1 で
  定義し、Task2(CLI)・Task4(テンプレート race.html)で同じキー名を使用。
  カード dict `{race_id,venue,number,name}` を Task3 で定義し Task4 で使用。
  `build_race_report(race_id)` / `today_cards()` / `fetch_race_cards(date)` の
  シグネチャは Task1/3 定義と Task4 利用で一致。
- **v1の割り切り**: レース名は未取得(`name=""`)で会場+R番号表示。仕様の
  「取れなければ空」に合致。
