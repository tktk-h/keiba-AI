# 全レース予想 + EV提案AI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 任意のJRAレースに対し、較正済みモデルの統計予想と、単勝・複勝・ワイド・馬連のEV提案を、各々確信度つきで提示する。

**Architecture:** 較正済み勝率(既存モデル)を起点に、ハーヴィル法で複勝・馬連・ワイド確率を導出。複勝/ワイド/馬連オッズをnetkeibaオッズAPIから取得し、確率×オッズでEVを算出。確信度(推定の信頼度)を付けて券種横断でEV順に提案する。純粋関数を先にTDDで作り、ネットワーク依存(オッズ取得)を最後に統合する。

**Tech Stack:** Python, pandas, scikit-learn(既存model.pkl), BeautifulSoup/requests(既存), pytest。

---

## File Structure

- `keiba/predictor.py`(変更): `place_k`、`quinella_probabilities`、`wide_probabilities` を追加。
- `keiba/expected_value.py`(変更): 汎用 `ev(prob, odds)` を追加。
- `keiba/confidence.py`(新規): 推定信頼度スコア。
- `keiba/recommend.py`(変更): `predict_ranking`(①予想)と `recommend_all`(②EV提案)を追加。
- `keiba/odds_page.py`(新規): 複勝/ワイド/馬連オッズの取得・パース。
- `main.py`(変更): ①予想 ②EV提案 を確信度つきで表示。
- テスト: `tests/test_predictor_combo.py`, `tests/test_confidence.py`, `tests/test_recommend_all.py`, `tests/test_odds_page.py`, 既存 `tests/` は壊さない。

既存の `predictor.py` には `win_probabilities(df, model)` と `place_probabilities(win, k=2)`(ハーヴィル法、順列ベース)が既にある。新関数はこの様式に合わせる。

---

## Task 1: 複勝の着内頭数ルール `place_k`

JRAルール: 出走8頭以上=3着以内、5〜7頭=2着以内、4頭以下=複勝なし。

**Files:**
- Modify: `keiba/predictor.py`
- Test: `tests/test_predictor_combo.py`

- [ ] **Step 1: 失敗するテストを書く**

```python
# tests/test_predictor_combo.py
from keiba.predictor import place_k


def test_place_k_rules():
    assert place_k(8) == 3
    assert place_k(18) == 3
    assert place_k(7) == 2
    assert place_k(5) == 2
    assert place_k(4) is None   # 4頭以下は複勝なし
    assert place_k(1) is None
```

- [ ] **Step 2: 失敗を確認**

Run: `PYTHONIOENCODING=utf-8 python -m pytest tests/test_predictor_combo.py::test_place_k_rules -v`
Expected: FAIL（`cannot import name 'place_k'`）

- [ ] **Step 3: 最小実装**

`keiba/predictor.py` の先頭 import 群の直後に追加:

```python
def place_k(field_size: int):
    """JRA複勝の着内頭数: 8頭以上=3, 5-7頭=2, 4頭以下=None(複勝なし)。"""
    if field_size >= 8:
        return 3
    if field_size >= 5:
        return 2
    return None
```

- [ ] **Step 4: テストパスを確認**

Run: `PYTHONIOENCODING=utf-8 python -m pytest tests/test_predictor_combo.py::test_place_k_rules -v`
Expected: PASS

- [ ] **Step 5: コミット**

```bash
git add keiba/predictor.py tests/test_predictor_combo.py
git commit -m "feat: add place_k (JRA 複勝 in-the-money rule)"
```

---

## Task 2: 馬連確率 `quinella_probabilities`(ハーヴィル法・順不同2頭が上位2着)

**Files:**
- Modify: `keiba/predictor.py`
- Test: `tests/test_predictor_combo.py`

- [ ] **Step 1: 失敗するテストを書く**

```python
# tests/test_predictor_combo.py に追記
from keiba.predictor import quinella_probabilities


def test_quinella_probabilities_basic():
    win = {"A": 0.5, "B": 0.3, "C": 0.2}
    q = quinella_probabilities(win)
    # キーは昇順タプル、3頭なら C(3,2)=3 ペア
    assert set(q.keys()) == {("A", "B"), ("A", "C"), ("B", "C")}
    # 全ペア確率の合計は 1.0(必ずどれか2頭が上位2着を占める)
    assert abs(sum(q.values()) - 1.0) < 1e-9
    # 手計算: P(A,B in top2) = (.5*.3)/(1-.5) + (.3*.5)/(1-.3)
    expected_ab = (0.5 * 0.3) / (1 - 0.5) + (0.3 * 0.5) / (1 - 0.3)
    assert abs(q[("A", "B")] - expected_ab) < 1e-9
    # 強い2頭(A,B)のペアが最も高い
    assert q[("A", "B")] == max(q.values())
```

- [ ] **Step 2: 失敗を確認**

Run: `PYTHONIOENCODING=utf-8 python -m pytest tests/test_predictor_combo.py::test_quinella_probabilities_basic -v`
Expected: FAIL（`cannot import name 'quinella_probabilities'`）

- [ ] **Step 3: 最小実装**

`keiba/predictor.py`（`from itertools import permutations` は既にある）に追加:

```python
def _candidate_names(win: dict, top_n: int):
    """確率上位 top_n 頭の名前(組み合わせ計算を現実的な規模に抑える)。"""
    ordered = sorted(win.items(), key=lambda kv: kv[1], reverse=True)
    return [name for name, _ in ordered[:top_n]]


def quinella_probabilities(win: dict, top_n: int = 12) -> dict:
    """馬連: 2頭がともに上位2着に入る確率(順不同)。キーは昇順タプル。

    ハーヴィル法。条件付き分母は全馬の勝率和 W を使う(上位 top_n のみ
    列挙するが、末尾の馬を含むペアは確率がほぼ0なので無視できる)。
    """
    names = _candidate_names(win, top_n)
    total = sum(win.values())
    out = {}
    for a, b in permutations(names, 2):
        p = (win[a] / total) * (win[b] / (total - win[a]))
        key = tuple(sorted((a, b)))
        out[key] = out.get(key, 0.0) + p
    return out
```

- [ ] **Step 4: テストパスを確認**

Run: `PYTHONIOENCODING=utf-8 python -m pytest tests/test_predictor_combo.py::test_quinella_probabilities_basic -v`
Expected: PASS

- [ ] **Step 5: コミット**

```bash
git add keiba/predictor.py tests/test_predictor_combo.py
git commit -m "feat: add quinella_probabilities (Harville pair-in-top2)"
```

---

## Task 3: ワイド確率 `wide_probabilities`(順不同2頭が上位3着)

**Files:**
- Modify: `keiba/predictor.py`
- Test: `tests/test_predictor_combo.py`

- [ ] **Step 1: 失敗するテストを書く**

```python
# tests/test_predictor_combo.py に追記
from keiba.predictor import wide_probabilities


def test_wide_ge_quinella():
    win = {"A": 0.4, "B": 0.3, "C": 0.2, "D": 0.1}
    q = quinella_probabilities(win)
    w = wide_probabilities(win)
    # ワイド(上位3着)はマージンが広いので、同じペアで馬連以上
    for pair in q:
        assert w[pair] >= q[pair] - 1e-12
    # 各確率は [0,1]
    for v in w.values():
        assert 0.0 <= v <= 1.0 + 1e-12
```

- [ ] **Step 2: 失敗を確認**

Run: `PYTHONIOENCODING=utf-8 python -m pytest tests/test_predictor_combo.py::test_wide_ge_quinella -v`
Expected: FAIL（`cannot import name 'wide_probabilities'`）

- [ ] **Step 3: 最小実装**

`keiba/predictor.py` に追加:

```python
def wide_probabilities(win: dict, top_n: int = 12) -> dict:
    """ワイド: 2頭がともに上位3着に入る確率(順不同)。キーは昇順タプル。

    上位3着の順列を列挙し、その集合に含まれる各ペアへ確率を加算する。
    """
    names = _candidate_names(win, top_n)
    total = sum(win.values())
    out = {}
    for a, b, c in permutations(names, 3):
        p = ((win[a] / total)
             * (win[b] / (total - win[a]))
             * (win[c] / (total - win[a] - win[b])))
        for x, y in ((a, b), (a, c), (b, c)):
            key = tuple(sorted((x, y)))
            out[key] = out.get(key, 0.0) + p
    return out
```

- [ ] **Step 4: テストパスを確認**

Run: `PYTHONIOENCODING=utf-8 python -m pytest tests/test_predictor_combo.py -v`
Expected: PASS（3テストとも）

- [ ] **Step 5: コミット**

```bash
git add keiba/predictor.py tests/test_predictor_combo.py
git commit -m "feat: add wide_probabilities (Harville pair-in-top3)"
```

---

## Task 4: 汎用EV関数 `ev`

**Files:**
- Modify: `keiba/expected_value.py`
- Test: `tests/test_expected_value.py`（無ければ新規）

- [ ] **Step 1: 失敗するテストを書く**

```python
# tests/test_expected_value.py
from keiba.expected_value import ev


def test_ev_break_even_and_sign():
    assert abs(ev(0.5, 2.0) - 0.0) < 1e-9      # 回収率100% = EV0
    assert ev(0.6, 2.0) > 0                      # +EV
    assert ev(0.4, 2.0) < 0                      # -EV
```

- [ ] **Step 2: 失敗を確認**

Run: `PYTHONIOENCODING=utf-8 python -m pytest tests/test_expected_value.py -v`
Expected: FAIL（`cannot import name 'ev'`）

- [ ] **Step 3: 最小実装**

`keiba/expected_value.py` の先頭に追加:

```python
def ev(prob: float, odds: float) -> float:
    """期待値 = 確率 × オッズ − 1(EV≥0 は回収率100%以上)。"""
    return prob * odds - 1.0
```

- [ ] **Step 4: テストパスを確認**

Run: `PYTHONIOENCODING=utf-8 python -m pytest tests/test_expected_value.py -v`
Expected: PASS

- [ ] **Step 5: コミット**

```bash
git add keiba/expected_value.py tests/test_expected_value.py
git commit -m "feat: add generic ev(prob, odds)"
```

---

## Task 5: 確信度 `confidence`

推定値そのものの信頼度(市場に勝てる自信ではない)。データ量・確率の鋭さ・オッズ確定・頭数の妥当性から 0–1 スコアと 低/中/高 を返す。

**Files:**
- Create: `keiba/confidence.py`
- Test: `tests/test_confidence.py`

- [ ] **Step 1: 失敗するテストを書く**

```python
# tests/test_confidence.py
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
```

- [ ] **Step 2: 失敗を確認**

Run: `PYTHONIOENCODING=utf-8 python -m pytest tests/test_confidence.py -v`
Expected: FAIL（`No module named 'keiba.confidence'`）

- [ ] **Step 3: 最小実装**

```python
# keiba/confidence.py
"""推定値の信頼度スコア。

「市場に勝てる自信」ではなく、その確率/EV推定をどれだけ信用してよいかを表す。
診断(docs/superpowers/specs/2026-06-15-ev-recommendation-ai-design.md 参照)で
モデルは較正済みだが市場超えはしないと分かっている。この信頼度はデータ品質に
基づく値であり、利益を約束するものではない。
"""


def confidence(data_runs: int, sharpness: float,
               odds_confirmed: bool, field_ok: bool):
    """0–1 のスコアと 低/中/高 を返す。

    data_runs:  関与馬の過去走数(連系は最小値)。多いほど高い(5で頭打ち)。
    sharpness:  抜けた本命の明確さ 0–1(混戦ほど低い)。
    odds_confirmed: 対象オッズが確定済み(None でない)か。
    field_ok:   出走頭数が妥当(5–18)か。
    """
    data = min(max(data_runs, 0), 5) / 5.0
    sharp = min(max(sharpness, 0.0), 1.0)
    odds = 1.0 if odds_confirmed else 0.5
    field = 1.0 if field_ok else 0.7
    score = 0.4 * data + 0.3 * sharp + 0.15 * odds + 0.15 * field
    score = round(score, 3)
    if score >= 0.66:
        level = "高"
    elif score >= 0.40:
        level = "中"
    else:
        level = "低"
    return score, level
```

- [ ] **Step 4: テストパスを確認**

Run: `PYTHONIOENCODING=utf-8 python -m pytest tests/test_confidence.py -v`
Expected: PASS（3テスト）

- [ ] **Step 5: コミット**

```bash
git add keiba/confidence.py tests/test_confidence.py
git commit -m "feat: add confidence score (estimate reliability, not market edge)"
```

---

## Task 6: ①予想ランキング `predict_ranking`

全馬を推定勝率順に、複勝率と確信度つきで返す。

**Files:**
- Modify: `keiba/recommend.py`
- Test: `tests/test_recommend_all.py`

`recommend.py` の既存 import は `from keiba.expected_value import win_ev`。新しい依存を追加する。

- [ ] **Step 1: 失敗するテストを書く**

```python
# tests/test_recommend_all.py
from keiba.models import Race, Horse, PastRun
from keiba.recommend import predict_ranking


def _horse(name, num, odds, n_runs):
    runs = [PastRun(date="2026-01-01", finish=1, course="東京", distance=1600,
                    surface="芝", track_condition="良", time=95.0, last_3f=33.5,
                    popularity=1, weight_carried=55.0, jockey="J",
                    race_class="G3", win_odds=3.0) for _ in range(n_runs)]
    return Horse(name=name, sex="牡", age=4, weight_carried=55.0, jockey="J",
                 post=num, number=num, win_odds=odds, popularity=num,
                 body_weight=480, body_weight_diff=0, running_style=None,
                 sire=None, dam=None, broodmare_sire=None, training_time=None,
                 training_course=None, training_eval=None, past_runs=runs)


def _race(n=8):
    horses = [_horse(chr(65 + i), i + 1, 2.0 + i, 3) for i in range(n)]
    return Race(race_id="r1", name="t", date="d", course="東京", distance=1600,
                surface="芝", turn="右", track_condition="良", weather="晴",
                horses=horses)


def test_predict_ranking_sorted_with_fields():
    race = _race(8)
    win = {h.name: (8 - i) / 36.0 for i, h in enumerate(race.horses)}  # A最強
    rows = predict_ranking(race, win)
    assert len(rows) == 8
    # 勝率降順
    probs = [r["win_prob"] for r in rows]
    assert probs == sorted(probs, reverse=True)
    top = rows[0]
    assert top["name"] == "A"
    assert 0.0 <= top["place_prob"] <= 1.0
    assert top["place_prob"] >= top["win_prob"]   # 複勝率≥勝率
    assert top["level"] in ("低", "中", "高")
```

- [ ] **Step 2: 失敗を確認**

Run: `PYTHONIOENCODING=utf-8 python -m pytest tests/test_recommend_all.py::test_predict_ranking_sorted_with_fields -v`
Expected: FAIL（`cannot import name 'predict_ranking'`）

- [ ] **Step 3: 最小実装**

`keiba/recommend.py` の import を次に置き換え、ヘルパとともに追加:

```python
from keiba.expected_value import win_ev, ev
from keiba.predictor import (place_k, place_probabilities,
                             quinella_probabilities, wide_probabilities)
from keiba.confidence import confidence


def _runs_by_name(race):
    return {h.name: len(h.past_runs) for h in race.horses}


def _race_sharpness(win_probs):
    ps = sorted(win_probs.values(), reverse=True)
    if not ps or ps[0] <= 0:
        return 0.0
    second = ps[1] if len(ps) > 1 else 0.0
    return max(0.0, min(1.0, (ps[0] - second) / ps[0]))


def _field_ok(race):
    return 5 <= len(race.horses) <= 18


def predict_ranking(race, win_probs: dict) -> list:
    """全馬を推定勝率順に、複勝率・確信度つきで返す(①予想)。"""
    runs = _runs_by_name(race)
    sharp = _race_sharpness(win_probs)
    fok = _field_ok(race)
    k = place_k(len(race.horses))
    place = place_probabilities(win_probs, k) if k else {}
    odds_ok = {h.name: (h.win_odds is not None) for h in race.horses}
    rows = []
    for name, p in sorted(win_probs.items(), key=lambda kv: kv[1], reverse=True):
        score, level = confidence(runs.get(name, 0), sharp,
                                   odds_ok.get(name, False), fok)
        rows.append({"name": name, "win_prob": p,
                     "place_prob": place.get(name), "confidence": score,
                     "level": level})
    return rows
```

注: `place_probabilities` は既存(`win, k`)。`k` が None(4頭以下)なら複勝率は空。

- [ ] **Step 4: テストパスを確認**

Run: `PYTHONIOENCODING=utf-8 python -m pytest tests/test_recommend_all.py::test_predict_ranking_sorted_with_fields -v`
Expected: PASS

- [ ] **Step 5: 既存テストの非破壊を確認**

Run: `PYTHONIOENCODING=utf-8 python -m pytest -q`
Expected: PASS（既存49 + 新規）

- [ ] **Step 6: コミット**

```bash
git add keiba/recommend.py tests/test_recommend_all.py
git commit -m "feat: add predict_ranking (per-race prediction with confidence)"
```

---

## Task 7: ②EV提案 `recommend_all`(券種横断ランキング)

単勝/複勝/ワイド/馬連の全候補をEV順に並べ、+EVの有無を返す。

**Files:**
- Modify: `keiba/recommend.py`
- Test: `tests/test_recommend_all.py`

オッズ引数の形:
```python
odds = {
    "place": {馬番: (下限, 上限)},
    "quinella": {(馬番a, 馬番b): オッズ},   # 馬番は任意順
    "wide": {(馬番a, 馬番b): オッズ},
}
```
(単勝オッズは `Race.horses[].win_odds` を使う)

- [ ] **Step 1: 失敗するテストを書く**

```python
# tests/test_recommend_all.py に追記
from keiba.recommend import recommend_all


def test_recommend_all_ranks_by_ev_and_flags_positive():
    race = _race(8)
    win = {h.name: (8 - i) / 36.0 for i, h in enumerate(race.horses)}
    # A(馬番1)の単勝を意図的に高オッズにして +EV を作る
    race.horses[0].win_odds = 99.0
    odds = {
        "place": {1: (1.5, 1.8)},
        "quinella": {(1, 2): 5.0},
        "wide": {(1, 2): 2.5},
    }
    bets, any_positive = recommend_all(race, win, odds, top_n=10)
    assert any_positive is True
    # EV降順
    evs = [b["ev"] for b in bets]
    assert evs == sorted(evs, reverse=True)
    # 4券種すべてが候補に含まれる
    assert {b["type"] for b in bets} == {"単勝", "複勝", "馬連", "ワイド"}
    for b in bets:
        assert b["level"] in ("低", "中", "高")


def test_recommend_all_negative_when_low_odds():
    race = _race(8)
    win = {h.name: (8 - i) / 36.0 for i, h in enumerate(race.horses)}
    for h in race.horses:
        h.win_odds = 1.1            # 全馬おいしくない単勝
    odds = {"place": {}, "quinella": {}, "wide": {}}
    bets, any_positive = recommend_all(race, win, odds, top_n=5)
    assert any_positive is False
    assert len(bets) >= 1           # それでも最良候補は返す(全レースで提示)
```

- [ ] **Step 2: 失敗を確認**

Run: `PYTHONIOENCODING=utf-8 python -m pytest tests/test_recommend_all.py::test_recommend_all_ranks_by_ev_and_flags_positive -v`
Expected: FAIL（`cannot import name 'recommend_all'`）

- [ ] **Step 3: 最小実装**

`keiba/recommend.py` に追加:

```python
def _num2name(race):
    return {h.number: h.name for h in race.horses}


def recommend_all(race, win_probs: dict, odds: dict, top_n: int = 8):
    """単勝/複勝/ワイド/馬連を券種横断でEV順に並べ、(bets, any_positive)を返す。

    +EV(ev>=0)が無くても、最良候補は必ず返す(全レースで予想を提示)。
    """
    runs = _runs_by_name(race)
    sharp = _race_sharpness(win_probs)
    fok = _field_ok(race)
    num2name = _num2name(race)
    k = place_k(len(race.horses))
    place = place_probabilities(win_probs, k) if k else {}
    quin = quinella_probabilities(win_probs)
    wide = wide_probabilities(win_probs)
    win_odds = {h.name: h.win_odds for h in race.horses}

    bets = []

    # 単勝
    for name, p in win_probs.items():
        o = win_odds.get(name)
        if o:
            cs, lv = confidence(runs.get(name, 0), sharp, True, fok)
            bets.append({"type": "単勝", "sel": name, "prob": p, "odds": o,
                         "ev": ev(p, o), "confidence": cs, "level": lv})

    # 複勝(下限オッズ採用=保守的)
    for num, lohi in odds.get("place", {}).items():
        name = num2name.get(num)
        p = place.get(name)
        if name is None or p is None:
            continue
        lo = lohi[0]
        cs, lv = confidence(runs.get(name, 0), sharp, True, fok)
        bets.append({"type": "複勝", "sel": name, "prob": p, "odds": lo,
                     "ev": ev(p, lo), "confidence": cs, "level": lv})

    # 馬連 / ワイド
    for kind, probmap, oddsmap in (("馬連", quin, odds.get("quinella", {})),
                                   ("ワイド", wide, odds.get("wide", {}))):
        for (na, nb), o in oddsmap.items():
            a, b = num2name.get(na), num2name.get(nb)
            if a is None or b is None:
                continue
            p = probmap.get(tuple(sorted((a, b))))
            if p is None:
                continue
            dr = min(runs.get(a, 0), runs.get(b, 0))
            cs, lv = confidence(dr, sharp, True, fok)
            bets.append({"type": kind, "sel": f"{na}-{nb}", "prob": p, "odds": o,
                         "ev": ev(p, o), "confidence": cs, "level": lv})

    bets.sort(key=lambda b: (b["ev"], b["confidence"]), reverse=True)
    any_positive = any(b["ev"] >= 0 for b in bets)
    return bets[:top_n], any_positive
```

- [ ] **Step 4: テストパスを確認**

Run: `PYTHONIOENCODING=utf-8 python -m pytest tests/test_recommend_all.py -v`
Expected: PASS（全テスト）

- [ ] **Step 5: 既存テストの非破壊を確認**

Run: `PYTHONIOENCODING=utf-8 python -m pytest -q`
Expected: PASS

- [ ] **Step 6: コミット**

```bash
git add keiba/recommend.py tests/test_recommend_all.py
git commit -m "feat: add recommend_all (cross-bet-type EV ranking with confidence)"
```

---

## Task 8: オッズ取得 `odds_page` — 実JSONをキャプチャしてパーサをTDD

netkeibaのオッズはオッズページのJSON APIから取得する。**JSON構造は実物を確認してから**パーサを書く(推測で書かない)。

**Files:**
- Create: `keiba/odds_page.py`
- Create: `tests/fixtures/odds_sample.json`（実データをキャプチャ)
- Test: `tests/test_odds_page.py`

- [ ] **Step 1: 実オッズJSONをキャプチャ(ネットワーク・手動)**

確定オッズのある直近レースIDで、複勝/ワイド/馬連のJSONを取得して保存する。
netkeibaのオッズAPIは `type` でプールを切り替える(例: 単勝=1, 複勝=1の別フィールド,
馬連=4, ワイド=5。実際の番号はレスポンスで確認)。まず1本取得して構造を見る:

```bash
RID=<直近の確定オッズのある race_id>
curl -s "https://race.netkeiba.com/api/api_get_jra_odds.html?race_id=${RID}&type=1&action=update" \
  -H "User-Agent: Mozilla/5.0" -o tests/fixtures/odds_tan_fuku.json
curl -s "https://race.netkeiba.com/api/api_get_jra_odds.html?race_id=${RID}&type=4&action=update" \
  -H "User-Agent: Mozilla/5.0" -o tests/fixtures/odds_umaren.json
curl -s "https://race.netkeiba.com/api/api_get_jra_odds.html?race_id=${RID}&type=5&action=update" \
  -H "User-Agent: Mozilla/5.0" -o tests/fixtures/odds_wide.json
```

`python -c "import json;print(json.load(open('tests/fixtures/odds_umaren.json',encoding='utf-8')))"`
で構造を確認し、馬番→オッズの格納場所(キー名)を特定する。確認した構造を
`tests/fixtures/odds_sample.json` に下記のフラットな形で1ファイルに**手で整形保存**する
(パーサのテスト入力を安定させるため):

```json
{
  "place": {"1": ["1.5", "1.8"], "2": ["2.1", "2.6"]},
  "quinella": {"1_2": "5.0", "1_3": "8.4"},
  "wide": {"1_2": "2.5", "1_3": "3.8"}
}
```

実APIのキー名/値配列の位置が分かったら、Step 3の `fetch_*` がこの整形済み形へ
変換するように合わせる(生レスポンス→この形)。

- [ ] **Step 2: 失敗するテストを書く**

```python
# tests/test_odds_page.py
import json
from pathlib import Path
from keiba.odds_page import parse_odds


def _sample():
    return json.loads(Path("tests/fixtures/odds_sample.json").read_text(encoding="utf-8"))


def test_parse_odds_shapes():
    odds = parse_odds(_sample())
    # 複勝: 馬番(int) -> (下限float, 上限float)
    assert odds["place"][1] == (1.5, 1.8)
    # 馬連/ワイド: 昇順(int,int)タプル -> float
    assert odds["quinella"][(1, 2)] == 5.0
    assert odds["wide"][(1, 2)] == 2.5
    # キーは昇順正規化
    assert all(a < b for (a, b) in odds["quinella"])
```

- [ ] **Step 3: 失敗を確認**

Run: `PYTHONIOENCODING=utf-8 python -m pytest tests/test_odds_page.py -v`
Expected: FAIL（`No module named 'keiba.odds_page'`）

- [ ] **Step 4: 最小実装**

```python
# keiba/odds_page.py
"""netkeibaオッズAPIから複勝・ワイド・馬連オッズを取得・整形する。

fetch_* はネットワーク(手動利用、テスト対象外)。parse_odds は
整形済みJSON(tests/fixtures/odds_sample.json の形)を Python の
扱いやすい dict へ変換する純粋関数で、テスト対象。
"""
import json
import requests

HEADERS = {"User-Agent": "Mozilla/5.0 (keiba-research)"}
API = "https://race.netkeiba.com/api/api_get_jra_odds.html"


def _pair_key(s: str):
    a, b = s.split("_")
    a, b = int(a), int(b)
    return (a, b) if a < b else (b, a)


def parse_odds(sample: dict) -> dict:
    """整形済みJSON -> {'place':{int:(lo,hi)}, 'quinella':{(a,b):float}, 'wide':...}。"""
    place = {int(k): (float(v[0]), float(v[1]))
             for k, v in sample.get("place", {}).items()}
    quinella = {_pair_key(k): float(v) for k, v in sample.get("quinella", {}).items()}
    wide = {_pair_key(k): float(v) for k, v in sample.get("wide", {}).items()}
    return {"place": place, "quinella": quinella, "wide": wide}


def _get(race_id: str, type_: int) -> dict:
    resp = requests.get(API, params={"race_id": race_id, "type": type_,
                                     "action": "update"}, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    return resp.json()


def fetch_odds(race_id: str) -> dict:
    """実APIから複勝/馬連/ワイドを取得し parse_odds の入力形へ整形する。

    注: 各 type の生レスポンス構造は Task 8 Step 1 で確認した形に合わせて
    ここで 'place'/'quinella'/'wide' のフラット形へ変換すること。
    """
    raw_tan_fuku = _get(race_id, 1)
    raw_umaren = _get(race_id, 4)
    raw_wide = _get(race_id, 5)
    sample = {
        "place": _extract_place(raw_tan_fuku),
        "quinella": _extract_pairs(raw_umaren),
        "wide": _extract_pairs(raw_wide),
    }
    return parse_odds(sample)


def _extract_place(raw: dict) -> dict:
    """生レスポンス -> {'馬番': ['下限','上限']}。Step 1 で確認したキーに合わせる。"""
    # 例: raw['data']['odds']['1'] が {馬番: ['min','max', ...]} の場合。
    block = raw.get("data", {}).get("odds", {})
    fuku = block.get("3") or block.get("1b") or {}
    return {str(num): [vals[0], vals[1]] for num, vals in fuku.items() if len(vals) >= 2}


def _extract_pairs(raw: dict) -> dict:
    """生レスポンス -> {'a_b': 'odds'}。Step 1 で確認したキーに合わせる。"""
    block = raw.get("data", {}).get("odds", {})
    inner = next(iter(block.values()), {}) if block else {}
    return {k: (v[0] if isinstance(v, list) else v) for k, v in inner.items()}
```

注: `_extract_place` / `_extract_pairs` は **Step 1 で確認した実構造に合わせて修正必須**。
`parse_odds`(テスト対象)は構造が安定しているので、ネットワーク部分の不確実性を
ここに閉じ込める設計。

- [ ] **Step 5: テストパスを確認**

Run: `PYTHONIOENCODING=utf-8 python -m pytest tests/test_odds_page.py -v`
Expected: PASS

- [ ] **Step 6: 実取得のスモーク確認(ネットワーク・手動)**

Run: `PYTHONIOENCODING=utf-8 python -c "from keiba.odds_page import fetch_odds; import pprint; pprint.pprint(fetch_odds('<RID>'))"`
Expected: place/quinella/wide が馬番→オッズで埋まる。空なら `_extract_*` のキーを実JSONに合わせて直す。

- [ ] **Step 7: コミット**

```bash
git add keiba/odds_page.py tests/test_odds_page.py tests/fixtures/odds_sample.json
git commit -m "feat: add odds_page (place/quinella/wide odds fetch + parse)"
```

---

## Task 9: CLI結線 `main.py`(①予想 ②EV提案 を確信度つきで表示)

**Files:**
- Modify: `main.py`
- 動作確認: 手動(ネットワーク)

- [ ] **Step 1: main.py を書き換える**

```python
import os
import sys
from keiba.scraper import scrape_race
from keiba.features import build_features
from keiba.predictor import win_probabilities
from keiba.recommend import predict_ranking, recommend_all
from keiba.odds_page import fetch_odds
from keiba.model import load_model, DEFAULT_MODEL_PATH


def main(race_id: str, enrich: bool = False):
    print(f"レース {race_id} を取得中...")
    if enrich:
        print("  各馬の過去成績・血統も取得します(時間がかかります)...")
    race = scrape_race(race_id, enrich=enrich)
    print(f"  {race.name} / {race.surface}{race.distance}m / {len(race.horses)}頭")

    df = build_features(race)
    model = None
    if os.path.exists(DEFAULT_MODEL_PATH):
        model = load_model(DEFAULT_MODEL_PATH)
        print("  学習済みモデルで予測します。")
    else:
        print("  モデル未学習のためオッズベースで予測します(train.pyで学習可)。")
    win_probs = win_probabilities(df, model=model)

    # ① 統計ベース予想(全馬)
    print("\n=== ① 予想(統計ベース・確信度つき) ===")
    for i, r in enumerate(predict_ranking(race, win_probs), 1):
        pp = f"{r['place_prob']:.1%}" if r["place_prob"] is not None else "—"
        print(f"{i:>2}. {r['name']:　<10} 勝率{r['win_prob']:5.1%} "
              f"複勝{pp:>6} 確信度{r['level']}({r['confidence']:.2f})")

    # ② EV提案(単勝/複勝/ワイド/馬連)
    try:
        odds = fetch_odds(race_id)
    except Exception as exc:  # noqa: BLE001
        print(f"\n(オッズ取得に失敗: {exc} — 単勝のみで提案します)")
        odds = {"place": {}, "quinella": {}, "wide": {}}
    bets, any_positive = recommend_all(race, win_probs, odds, top_n=8)
    print("\n=== ② EV提案(期待値の高い買い方) ===")
    if not bets:
        print("(オッズ未確定のため計算できません。レース直前のIDを指定してください)")
        return
    if not any_positive:
        print("※ +EVの買い目はありません。今レースは見送り推奨(以下は参考)。")
    for i, b in enumerate(bets, 1):
        mark = "＋" if b["ev"] >= 0 else "－"
        print(f"{i}. [{b['type']}] {b['sel']:　<10} オッズ{b['odds']:>6} "
              f"推定{b['prob']:5.1%} 期待値{b['ev']:+.2f}{mark} "
              f"確信度{b['level']}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    enrich = "--enrich" in sys.argv
    if not args:
        print("使い方: python main.py <race_id> [--enrich]")
        sys.exit(1)
    main(args[0], enrich=enrich)
```

- [ ] **Step 2: 構文・import確認**

Run: `PYTHONIOENCODING=utf-8 python -c "import main"`
Expected: エラーなし

- [ ] **Step 3: 実レースでスモーク確認(ネットワーク・手動)**

Run: `PYTHONIOENCODING=utf-8 python main.py <確定オッズのある race_id> --enrich`
Expected: ①予想(全馬・確信度)と ②EV提案(券種横断・確信度・+EV有無)が表示される。

- [ ] **Step 4: コミット**

```bash
git add main.py
git commit -m "feat: wire prediction + EV recommendation into main CLI"
```

---

## Task 10: 全体テストと仕上げ

- [ ] **Step 1: 全テスト実行**

Run: `PYTHONIOENCODING=utf-8 python -m pytest -q`
Expected: 既存49 + 新規(predictor_combo 3, expected_value 1, confidence 3, recommend_all 3, odds_page 1)= 60 前後すべてPASS

- [ ] **Step 2: 一時診断スクリプトの整理**

`diagnose_edge.py` / `diagnose_noodds.py` は調査用。残す場合は `docs/` 配下か
コメントで用途を明記、不要なら削除する(ユーザに確認)。

- [ ] **Step 3: 最終コミット(必要なら)**

```bash
git add -A
git commit -m "chore: finalize EV recommendation AI v1"
```

---

## Self-Review(著者チェック済み)

- **仕様カバレッジ**: ①予想=Task6、②EV提案(単勝/複勝/ワイド/馬連)=Task7、
  複勝kルール=Task1、馬連/ワイド確率=Task2/3、EV=Task4、確信度=Task5、
  オッズ取得=Task8、CLI=Task9。スコープ外(3連系・ケリー)は未実装で正しい。
- **プレースホルダ**: ネットワーク部(Task8 `_extract_*`、Task9 スモーク)は
  実データ確認が必要な性質のため「実構造に合わせて修正」と明記。純粋関数側は
  完全コードを記載。
- **型整合**: `confidence(...) -> (score, level)`、`recommend_all -> (bets, any_positive)`、
  オッズ形 `{'place':{int:(lo,hi)},'quinella':{(int,int):float},'wide':...}` を
  Task間で一貫使用。確率はすべて name キー、オッズは馬番キーで、`recommend_all`
  内の `_num2name` で対応付け。
