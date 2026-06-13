# 中央競馬 期待値予想プログラム 実装計画 (v1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** netkeibaから取得した出走馬・過去成績・血統・調教・オッズデータをもとに、ロジスティック回帰で各馬の確率を推定し、複数券種の期待値を計算して期待値の高い買い方をランキング表示するプログラムを作る。

**Architecture:** 5つのモジュール(データ収集 / 特徴量作成 / 予測 / 期待値計算 / 推奨出力)に分割。各モジュールは独立してテスト可能。データ収集は生データをJSONで保存し、後続モジュールはそのデータ構造を入力として受け取る。モデルは差し替え可能なインターフェースにし、v1はロジスティック回帰を実装する。

**Tech Stack:** Python 3.11+, requests, beautifulsoup4, pandas, scikit-learn, pytest

---

## File Structure

```
競馬/
├── keiba/
│   ├── __init__.py
│   ├── models.py            # データクラス(Race, Horse, PastRun など)
│   ├── scraper.py           # netkeibaスクレイピング
│   ├── storage.py           # JSON保存/読み込み
│   ├── features.py          # 特徴量作成
│   ├── predictor.py         # 確率推定(ロジスティック回帰 + ハーヴィル法)
│   ├── expected_value.py    # 期待値計算
│   └── recommend.py         # 推奨出力
├── tests/
│   ├── __init__.py
│   ├── fixtures/            # サンプルHTML/JSON
│   ├── test_models.py
│   ├── test_scraper.py
│   ├── test_storage.py
│   ├── test_features.py
│   ├── test_predictor.py
│   ├── test_expected_value.py
│   └── test_recommend.py
├── main.py                  # エントリポイント(レースID指定 → 推奨出力)
├── requirements.txt
└── README.md
```

---

## Task 1: プロジェクト初期化

**Files:**
- Create: `requirements.txt`
- Create: `keiba/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: requirements.txt を作成**

```
requests>=2.31
beautifulsoup4>=4.12
pandas>=2.0
scikit-learn>=1.3
pytest>=7.4
```

- [ ] **Step 2: 空の __init__.py を作成**

`keiba/__init__.py` と `tests/__init__.py` を空ファイルで作成する。

- [ ] **Step 3: 依存をインストール**

Run: `pip install -r requirements.txt`
Expected: 全パッケージが正常にインストールされる

- [ ] **Step 4: pytestが動くことを確認**

Run: `pytest -q`
Expected: "no tests ran"(エラーなし)

- [ ] **Step 5: Commit**

```bash
git add requirements.txt keiba/__init__.py tests/__init__.py
git commit -m "chore: project init with dependencies"
```

---

## Task 2: データモデル定義

出走馬・過去成績・レースを表すデータクラスを定義する。全モジュールが共有する。

**Files:**
- Create: `keiba/models.py`
- Test: `tests/test_models.py`

- [ ] **Step 1: 失敗するテストを書く**

```python
# tests/test_models.py
from keiba.models import PastRun, Horse, Race

def test_horse_holds_basic_and_past_runs():
    past = PastRun(date="2026-05-01", finish=1, course="東京", distance=1600,
                   surface="芝", track_condition="良", time=95.2, last_3f=33.5,
                   popularity=2, weight_carried=55.0, jockey="ルメール", race_class="G3")
    horse = Horse(name="テスト馬", sex="牡", age=4, weight_carried=56.0,
                  jockey="ルメール", post=3, number=5, win_odds=4.2, popularity=2,
                  body_weight=480, body_weight_diff=2, running_style="先行",
                  sire="ディープインパクト", dam="テスト母", broodmare_sire="母父馬",
                  training_time=None, training_course=None, training_eval=None,
                  past_runs=[past])
    assert horse.past_runs[0].finish == 1
    assert horse.win_odds == 4.2

def test_race_holds_horses():
    race = Race(race_id="202605010101", name="テストS", date="2026-05-01",
                course="東京", distance=1600, surface="芝", turn="右",
                track_condition="良", weather="晴", horses=[])
    assert race.race_id == "202605010101"
    assert race.horses == []
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `pytest tests/test_models.py -v`
Expected: FAIL("No module named 'keiba.models'")

- [ ] **Step 3: models.py を実装**

```python
# keiba/models.py
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class PastRun:
    date: str
    finish: int
    course: str
    distance: int
    surface: str          # 芝/ダート
    track_condition: str  # 良/稍重/重/不良
    time: Optional[float]      # 走破時計(秒)
    last_3f: Optional[float]   # 上がり3F(秒)
    popularity: Optional[int]
    weight_carried: Optional[float]
    jockey: Optional[str]
    race_class: Optional[str]

@dataclass
class Horse:
    name: str
    sex: str
    age: int
    weight_carried: float
    jockey: str
    post: int             # 枠番
    number: int           # 馬番
    win_odds: Optional[float]
    popularity: Optional[int]
    body_weight: Optional[int]
    body_weight_diff: Optional[int]
    running_style: Optional[str]
    sire: Optional[str]
    dam: Optional[str]
    broodmare_sire: Optional[str]
    training_time: Optional[str]
    training_course: Optional[str]
    training_eval: Optional[str]
    past_runs: list = field(default_factory=list)

@dataclass
class Race:
    race_id: str
    name: str
    date: str
    course: str
    distance: int
    surface: str
    turn: str             # 右/左
    track_condition: str
    weather: str
    horses: list = field(default_factory=list)
```

- [ ] **Step 4: テストが通ることを確認**

Run: `pytest tests/test_models.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add keiba/models.py tests/test_models.py
git commit -m "feat: add data models for race, horse, past run"
```

---

## Task 3: ストレージ(JSON保存/読み込み)

収集したRaceデータをJSONで保存し、読み戻せるようにする。

**Files:**
- Create: `keiba/storage.py`
- Test: `tests/test_storage.py`

- [ ] **Step 1: 失敗するテストを書く**

```python
# tests/test_storage.py
from keiba.models import Race, Horse, PastRun
from keiba.storage import save_race, load_race

def _sample_race():
    past = PastRun(date="2026-05-01", finish=1, course="東京", distance=1600,
                   surface="芝", track_condition="良", time=95.2, last_3f=33.5,
                   popularity=2, weight_carried=55.0, jockey="ルメール", race_class="G3")
    horse = Horse(name="テスト馬", sex="牡", age=4, weight_carried=56.0,
                  jockey="ルメール", post=3, number=5, win_odds=4.2, popularity=2,
                  body_weight=480, body_weight_diff=2, running_style="先行",
                  sire="ディープ", dam="母", broodmare_sire="母父",
                  training_time=None, training_course=None, training_eval=None,
                  past_runs=[past])
    return Race(race_id="202605010101", name="テストS", date="2026-05-01",
                course="東京", distance=1600, surface="芝", turn="右",
                track_condition="良", weather="晴", horses=[horse])

def test_save_and_load_roundtrip(tmp_path):
    race = _sample_race()
    path = tmp_path / "race.json"
    save_race(race, path)
    loaded = load_race(path)
    assert loaded.race_id == race.race_id
    assert loaded.horses[0].name == "テスト馬"
    assert loaded.horses[0].past_runs[0].finish == 1
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `pytest tests/test_storage.py -v`
Expected: FAIL("No module named 'keiba.storage'")

- [ ] **Step 3: storage.py を実装**

```python
# keiba/storage.py
import json
from dataclasses import asdict
from pathlib import Path
from keiba.models import Race, Horse, PastRun

def save_race(race: Race, path) -> None:
    Path(path).write_text(
        json.dumps(asdict(race), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

def load_race(path) -> Race:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    horses = []
    for h in data["horses"]:
        past_runs = [PastRun(**pr) for pr in h.pop("past_runs", [])]
        horses.append(Horse(past_runs=past_runs, **h))
    data["horses"] = horses
    return Race(**data)
```

- [ ] **Step 4: テストが通ることを確認**

Run: `pytest tests/test_storage.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add keiba/storage.py tests/test_storage.py
git commit -m "feat: add JSON storage for race data"
```

---

## Task 4: スクレイパー(パース部分)

netkeibaのレース結果ページHTMLからRaceデータを組み立てる。ネットワーク部分とパース部分を分離し、パースをテスト可能にする。

**Files:**
- Create: `keiba/scraper.py`
- Create: `tests/fixtures/race_sample.html`
- Test: `tests/test_scraper.py`

- [ ] **Step 1: フィクスチャHTMLを用意**

実際のnetkeibaレースページの構造を反映した最小HTMLを `tests/fixtures/race_sample.html` に作成する。出馬表テーブル(馬番・馬名・性齢・斤量・騎手・単勝オッズ・人気・馬体重)を1〜2頭分含める。

> 注: 実装者は実際のnetkeibaページ(例 `https://race.netkeiba.com/race/shutuba.html?race_id=...`)のHTMLを取得し、テーブルのclass名・列構成を確認してから、それに合わせた最小フィクスチャを作ること。

- [ ] **Step 2: 失敗するテストを書く**

```python
# tests/test_scraper.py
from pathlib import Path
from keiba.scraper import parse_race

def test_parse_race_extracts_horses():
    html = Path("tests/fixtures/race_sample.html").read_text(encoding="utf-8")
    race = parse_race(html, race_id="202605010101")
    assert race.race_id == "202605010101"
    assert len(race.horses) >= 1
    first = race.horses[0]
    assert first.name
    assert first.number >= 1
```

- [ ] **Step 3: テストが失敗することを確認**

Run: `pytest tests/test_scraper.py -v`
Expected: FAIL("No module named 'keiba.scraper'")

- [ ] **Step 4: scraper.py を実装**

```python
# keiba/scraper.py
import requests
from bs4 import BeautifulSoup
from keiba.models import Race, Horse

HEADERS = {"User-Agent": "Mozilla/5.0 (keiba-research)"}

def fetch_html(url: str) -> str:
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding
    return resp.text

def parse_race(html: str, race_id: str) -> Race:
    soup = BeautifulSoup(html, "html.parser")
    # 実装者注: 実際のページ構造に合わせてセレクタを調整すること。
    # 以下はフィクスチャ構造に対応する最小実装の雛形。
    horses = []
    for row in soup.select("table.shutuba tr.horse"):
        cells = [c.get_text(strip=True) for c in row.select("td")]
        # cells の並びはフィクスチャに合わせる
        horses.append(Horse(
            name=cells[1], sex=cells[2][0], age=int(cells[2][1:]),
            weight_carried=float(cells[3]), jockey=cells[4],
            post=int(cells[0]), number=int(cells[0]),
            win_odds=float(cells[5]) if cells[5] else None,
            popularity=int(cells[6]) if len(cells) > 6 and cells[6] else None,
            body_weight=None, body_weight_diff=None, running_style=None,
            sire=None, dam=None, broodmare_sire=None,
            training_time=None, training_course=None, training_eval=None,
            past_runs=[],
        ))
    return Race(race_id=race_id, name="", date="", course="", distance=0,
                surface="", turn="", track_condition="", weather="", horses=horses)

def scrape_race(race_id: str) -> Race:
    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
    return parse_race(fetch_html(url), race_id)
```

- [ ] **Step 5: テストが通ることを確認**

Run: `pytest tests/test_scraper.py -v`
Expected: PASS (1 passed)

> 注: fetch_html / scrape_race(ネットワーク)はv1では自動テスト対象外。手動で実レースIDを指定して動作確認する。過去成績・血統・調教の取得は Task 4b として段階的に追加する(各馬ページ `https://db.netkeiba.com/horse/<id>` をパースし past_runs・血統を埋める)。

- [ ] **Step 6: Commit**

```bash
git add keiba/scraper.py tests/test_scraper.py tests/fixtures/race_sample.html
git commit -m "feat: add netkeiba race page scraper with parse tests"
```

---

## Task 5: 特徴量作成

Horseのデータから予測モデル用の数値特徴量(pandas DataFrame)を作る。

**Files:**
- Create: `keiba/features.py`
- Test: `tests/test_features.py`

- [ ] **Step 1: 失敗するテストを書く**

```python
# tests/test_features.py
from keiba.models import Race, Horse, PastRun
from keiba.features import build_features

def _horse(name, odds, finishes):
    runs = [PastRun(date="2026-05-01", finish=f, course="東京", distance=1600,
                    surface="芝", track_condition="良", time=95.0, last_3f=33.5,
                    popularity=1, weight_carried=55.0, jockey="J", race_class="G3")
            for f in finishes]
    return Horse(name=name, sex="牡", age=4, weight_carried=56.0, jockey="J",
                 post=1, number=1, win_odds=odds, popularity=1, body_weight=480,
                 body_weight_diff=0, running_style=None, sire=None, dam=None,
                 broodmare_sire=None, training_time=None, training_course=None,
                 training_eval=None, past_runs=runs)

def test_build_features_columns_and_rows():
    race = Race(race_id="r1", name="t", date="d", course="東京", distance=1600,
                surface="芝", turn="右", track_condition="良", weather="晴",
                horses=[_horse("A", 2.0, [1, 2, 1]), _horse("B", 10.0, [5, 8, 6])])
    df = build_features(race)
    assert len(df) == 2
    assert "win_odds" in df.columns
    assert "avg_finish" in df.columns
    # Aの平均着順 < Bの平均着順
    a = df[df["name"] == "A"].iloc[0]
    b = df[df["name"] == "B"].iloc[0]
    assert a["avg_finish"] < b["avg_finish"]
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `pytest tests/test_features.py -v`
Expected: FAIL("No module named 'keiba.features'")

- [ ] **Step 3: features.py を実装**

```python
# keiba/features.py
import pandas as pd
from keiba.models import Race

def _avg(values):
    nums = [v for v in values if v is not None]
    return sum(nums) / len(nums) if nums else None

def build_features(race: Race) -> pd.DataFrame:
    rows = []
    for h in race.horses:
        finishes = [r.finish for r in h.past_runs]
        last3f = [r.last_3f for r in h.past_runs]
        rows.append({
            "name": h.name,
            "number": h.number,
            "win_odds": h.win_odds,
            "age": h.age,
            "weight_carried": h.weight_carried,
            "avg_finish": _avg(finishes),
            "best_finish": min(finishes) if finishes else None,
            "avg_last_3f": _avg(last3f),
            "n_past_runs": len(h.past_runs),
        })
    return pd.DataFrame(rows)
```

- [ ] **Step 4: テストが通ることを確認**

Run: `pytest tests/test_features.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add keiba/features.py tests/test_features.py
git commit -m "feat: add feature engineering from race data"
```

---

## Task 6: 予測モジュール(確率推定)

学習済みロジスティック回帰で各馬の1着確率を出し、正規化して合計1にする。2着・3着確率はハーヴィル法で近似。v1では学習データが無い場合に備え、オッズベースの暫定確率にフォールバックする。

**Files:**
- Create: `keiba/predictor.py`
- Test: `tests/test_predictor.py`

- [ ] **Step 1: 失敗するテストを書く**

```python
# tests/test_predictor.py
import pandas as pd
from keiba.predictor import win_probabilities, place_probabilities

def test_win_probabilities_sum_to_one():
    df = pd.DataFrame({"name": ["A", "B", "C"], "win_odds": [2.0, 4.0, 8.0]})
    probs = win_probabilities(df)
    assert abs(sum(probs.values()) - 1.0) < 1e-6
    # オッズが低い馬ほど確率が高い
    assert probs["A"] > probs["B"] > probs["C"]

def test_place_probabilities_harville():
    win = {"A": 0.5, "B": 0.3, "C": 0.2}
    place = place_probabilities(win, k=2)  # 2着以内に入る確率
    assert place["A"] > place["B"] > place["C"]
    assert all(0 <= p <= 1 for p in place.values())
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `pytest tests/test_predictor.py -v`
Expected: FAIL("No module named 'keiba.predictor'")

- [ ] **Step 3: predictor.py を実装**

```python
# keiba/predictor.py
from itertools import permutations

def win_probabilities(df) -> dict:
    """オッズベースの暫定勝率(1/odds を正規化)。
    学習済みモデルがある場合は後でここを差し替える。"""
    raw = {}
    for _, row in df.iterrows():
        odds = row.get("win_odds")
        raw[row["name"]] = (1.0 / odds) if odds and odds > 0 else 0.0
    total = sum(raw.values())
    if total == 0:
        n = len(raw)
        return {k: 1.0 / n for k in raw}
    return {k: v / total for k, v in raw.items()}

def place_probabilities(win: dict, k: int = 2) -> dict:
    """ハーヴィル法: k着以内に入る確率を勝率から近似算出。"""
    names = list(win.keys())
    result = {n: 0.0 for n in names}
    for order in permutations(names, k):
        remaining = dict(win)
        p = 1.0
        for horse in order:
            denom = sum(remaining.values())
            if denom == 0:
                p = 0.0
                break
            p *= remaining[horse] / denom
            del remaining[horse]
        for horse in order:
            result[horse] += p
    return result
```

- [ ] **Step 4: テストが通ることを確認**

Run: `pytest tests/test_predictor.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add keiba/predictor.py tests/test_predictor.py
git commit -m "feat: add win/place probability estimation"
```

---

## Task 7: 期待値計算

推定確率と各券種オッズから期待値を計算する。

**Files:**
- Create: `keiba/expected_value.py`
- Test: `tests/test_expected_value.py`

- [ ] **Step 1: 失敗するテストを書く**

```python
# tests/test_expected_value.py
from keiba.expected_value import win_ev, place_ev

def test_win_ev_positive_when_underpriced():
    # 確率0.5、オッズ3.0 → EV = 0.5*3.0 - 1 = 0.5
    assert abs(win_ev(prob=0.5, odds=3.0) - 0.5) < 1e-9

def test_win_ev_negative_when_overpriced():
    # 確率0.2、オッズ3.0 → EV = -0.4
    assert win_ev(prob=0.2, odds=3.0) < 0

def test_place_ev():
    assert abs(place_ev(prob=0.6, odds=2.0) - 0.2) < 1e-9
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `pytest tests/test_expected_value.py -v`
Expected: FAIL("No module named 'keiba.expected_value'")

- [ ] **Step 3: expected_value.py を実装**

```python
# keiba/expected_value.py

def win_ev(prob: float, odds: float) -> float:
    """単勝期待値: 確率×オッズ - 1。0なら損益なし、正なら期待値プラス。"""
    return prob * odds - 1.0

def place_ev(prob: float, odds: float) -> float:
    """複勝期待値(複勝オッズを使用)。"""
    return prob * odds - 1.0

def combo_prob(probs: list) -> float:
    """組み合わせ確率の簡易計算(独立性を仮定して積を取る)。
    馬連・三連複など複数頭がからむ券種の暫定確率。"""
    result = 1.0
    for p in probs:
        result *= p
    return result
```

- [ ] **Step 4: テストが通ることを確認**

Run: `pytest tests/test_expected_value.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add keiba/expected_value.py tests/test_expected_value.py
git commit -m "feat: add expected value calculation"
```

---

## Task 8: 推奨出力

各馬の単勝・複勝の期待値を計算し、期待値の高い順にランキングして返す。

**Files:**
- Create: `keiba/recommend.py`
- Test: `tests/test_recommend.py`

- [ ] **Step 1: 失敗するテストを書く**

```python
# tests/test_recommend.py
import pandas as pd
from keiba.recommend import recommend_bets

def test_recommend_sorted_by_ev():
    df = pd.DataFrame({"name": ["A", "B"], "win_odds": [3.0, 10.0]})
    win_probs = {"A": 0.5, "B": 0.1}
    recs = recommend_bets(df, win_probs, top_n=2)
    assert recs[0]["ev"] >= recs[1]["ev"]
    assert recs[0]["bet_type"] == "単勝"
    assert "name" in recs[0]
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `pytest tests/test_recommend.py -v`
Expected: FAIL("No module named 'keiba.recommend'")

- [ ] **Step 3: recommend.py を実装**

```python
# keiba/recommend.py
from keiba.expected_value import win_ev

def recommend_bets(df, win_probs: dict, top_n: int = 5) -> list:
    """単勝の期待値を計算し、期待値の高い順に上位top_n件を返す。"""
    bets = []
    for _, row in df.iterrows():
        name = row["name"]
        odds = row.get("win_odds")
        prob = win_probs.get(name)
        if odds and prob is not None:
            bets.append({
                "bet_type": "単勝",
                "name": name,
                "odds": odds,
                "prob": prob,
                "ev": win_ev(prob, odds),
            })
    bets.sort(key=lambda b: b["ev"], reverse=True)
    return bets[:top_n]
```

- [ ] **Step 4: テストが通ることを確認**

Run: `pytest tests/test_recommend.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add keiba/recommend.py tests/test_recommend.py
git commit -m "feat: add bet recommendation ranking"
```

---

## Task 9: エントリポイント(main.py)

レースIDを引数で受け取り、収集→特徴量→確率→推奨までを通して実行しコンソール出力する。

**Files:**
- Create: `main.py`
- Create: `README.md`

- [ ] **Step 1: main.py を実装**

```python
# main.py
import sys
from keiba.scraper import scrape_race
from keiba.features import build_features
from keiba.predictor import win_probabilities
from keiba.recommend import recommend_bets

def main(race_id: str):
    print(f"レース {race_id} を取得中...")
    race = scrape_race(race_id)
    df = build_features(race)
    win_probs = win_probabilities(df)
    recs = recommend_bets(df, win_probs, top_n=5)
    print("\n=== 期待値の高い買い方 TOP5 ===")
    for i, b in enumerate(recs, 1):
        print(f"{i}. {b['bet_type']} {b['name']} "
              f"オッズ{b['odds']} 推定確率{b['prob']:.1%} 期待値{b['ev']:+.2f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使い方: python main.py <race_id>")
        sys.exit(1)
    main(sys.argv[1])
```

- [ ] **Step 2: 手動で実レースIDを指定して動作確認**

Run: `python main.py 202605010101`(実在のレースIDに置き換える)
Expected: 取得して TOP5 が表示される。スクレイパーのセレクタはTask 4の注記どおり実ページに合わせて調整が必要。

- [ ] **Step 3: README.md を作成**

```markdown
# 中央競馬 期待値予想プログラム

netkeibaのデータをもとに各馬の確率を推定し、期待値の高い買い方を提案する。

## セットアップ
pip install -r requirements.txt

## 使い方
python main.py <race_id>

## 構成
keiba/ 配下に収集・特徴量・予測・期待値・推奨の各モジュール。
詳細は docs/superpowers/specs/ と docs/superpowers/plans/ を参照。

## 注意
個人の分析目的での利用。スクレイピング時はアクセス間隔を空け、対象サイトの利用規約に従うこと。
```

- [ ] **Step 4: 全テストが通ることを確認**

Run: `pytest -q`
Expected: 全テスト PASS

- [ ] **Step 5: Commit**

```bash
git add main.py README.md
git commit -m "feat: add entrypoint and README"
```

---

## 今後の拡張(v2以降)
- 過去成績・血統・調教データの取得実装(Task 4b)と特徴量への組み込み
- ロジスティック回帰/LightGBM の学習(蓄積データ使用)、win_probabilities の差し替え
- 馬連・三連複など複数券種の期待値計算と組み合わせ最適化
- バックテスト(的中率・回収率の検証)
- アクセス間隔制御・キャッシュ・robots尊重などスクレイピングのマナー強化
