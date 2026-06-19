# 予想ダッシュボード Phase 1 (共有ロジック + CLI) 実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** モデルの予想勝率・市場との乖離スコア(-5..+5)・乖離馬の非オッズ根拠を、CLI(main.py)に「予想の資料」として表示する。ロジックは共有レイヤに置き、後のスマホWeb(Phase 2)がそのまま再利用できる形にする。

**Architecture:** 純粋関数を2箇所に追加する。(1) `keiba/recommend.py` に `deviation_score`(勝率の対数比→±5)、`predict_ranking` の各行へ `score` を付与。(2) 新規 `keiba/explain.py` にロジスティック回帰の線形寄与アトリビューション(`feature_contributions`)と、乖離馬の根拠抽出(`deviation_reasons`/`attach_reasons`)。`assemble_report` が `model`/`features` を受け取れば乖離馬に `reasons` を付ける。`main.py` は `report` dict を描画するだけ。

**Tech Stack:** Python, scikit-learn (Pipeline: StandardScaler + LogisticRegression), pandas, pytest.

**Scope:** Phase 1 のみ。Phase 2 (web/templates/race.html のスマホ対応) と Phase 3 (公開方法) は別計画。

---

### Task 1: `deviation_score` (乖離スコア -5..+5)

**Files:**
- Modify: `keiba/recommend.py`
- Test: `tests/test_recommend.py`

- [ ] **Step 1: Write the failing test**

`tests/test_recommend.py` の末尾に追記:

```python
from keiba.recommend import deviation_score


def test_deviation_score_even_is_zero():
    assert deviation_score(0.3, 0.3) == 0


def test_deviation_score_undervalued_is_positive():
    # モデルが市場の1.3倍 -> +1 (市場が過小評価)
    assert deviation_score(0.39, 0.30) == 1


def test_deviation_score_overvalued_is_negative():
    # モデルが市場の 1/1.3 -> -1 (市場が過大評価)
    assert deviation_score(0.30, 0.39) == -1


def test_deviation_score_clamped_to_five():
    assert deviation_score(0.99, 0.01) == 5
    assert deviation_score(0.01, 0.99) == -5


def test_deviation_score_zero_when_missing():
    assert deviation_score(0.0, 0.3) == 0
    assert deviation_score(0.3, 0.0) == 0
    assert deviation_score(None, 0.3) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_recommend.py -q`
Expected: FAIL — `ImportError: cannot import name 'deviation_score'`

- [ ] **Step 3: Write minimal implementation**

`keiba/recommend.py` の `VALUE_FACTOR = 1.3` 定義の直後に追加:

```python
def deviation_score(model_p: float, market_p: float) -> int:
    """モデル勝率と市場勝率の乖離を -5..+5 の11段階で返す。

    + は市場が過小評価(モデルが高評価=妙味)、- は過大評価(過剰人気)、0は互角。
    score = clamp(round(ln(model_p/market_p) / ln(VALUE_FACTOR)), -5, +5)。
    どちらかが 0/None なら評価不能として 0。
    """
    if not model_p or not market_p or model_p <= 0 or market_p <= 0:
        return 0
    raw = math.log(model_p / market_p) / math.log(VALUE_FACTOR)
    return max(-5, min(5, round(raw)))
```

(`keiba/recommend.py` は冒頭で既に `import math` 済み。)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_recommend.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add keiba/recommend.py tests/test_recommend.py
git commit -m "feat: deviation_score (-5..+5 market mispricing scale)"
```

---

### Task 2: `predict_ranking` の各行に `score` を付与

**Files:**
- Modify: `keiba/recommend.py` (`predict_ranking`)
- Test: `tests/test_report.py`

- [ ] **Step 1: Write the failing test**

`tests/test_report.py` の `test_assemble_report_structure` の末尾(関数内)に追記:

```python
    assert all("score" in p for p in rep["predictions"])   # 乖離スコア
    assert all(-5 <= p["score"] <= 5 for p in rep["predictions"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_report.py::test_assemble_report_structure -q`
Expected: FAIL — `KeyError: 'score'` / assertion error

- [ ] **Step 3: Write minimal implementation**

`keiba/recommend.py` の `predict_ranking` 内、`rows.append({...})` の辞書に `"score"` を追加:

```python
        rows.append({"name": name, "win_prob": p,
                     "place_prob": place.get(name), "market_prob": mp,
                     "value": _value_tag(p, mp),
                     "score": deviation_score(p, mp),
                     "confidence": score, "level": level})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_report.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add keiba/recommend.py tests/test_report.py
git commit -m "feat: attach deviation score to each prediction row"
```

---

### Task 3: `keiba/explain.py` — `feature_contributions`

**Files:**
- Create: `keiba/explain.py`
- Test: `tests/test_explain.py`

- [ ] **Step 1: Write the failing test**

`tests/test_explain.py` を新規作成:

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keiba.explain import feature_contributions


def _model():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [4.0, 3.0, 2.0, 1.0]})
    y = [0, 0, 1, 1]
    m = Pipeline([("scaler", StandardScaler()),
                  ("clf", LogisticRegression(max_iter=1000))])
    m.fit(X, y)
    m.feature_columns_ = ["a", "b"]
    return m


def test_feature_contributions_matches_coef_times_z():
    m = _model()
    row = {"a": 3.0, "b": 2.0}
    c = feature_contributions(m, row)
    sc = m.named_steps["scaler"]
    cl = m.named_steps["clf"]
    for i, k in enumerate(["a", "b"]):
        z = (row[k] - sc.mean_[i]) / sc.scale_[i]
        assert abs(c[k] - cl.coef_[0][i] * z) < 1e-9


def test_feature_contributions_missing_is_zero():
    m = _model()
    c = feature_contributions(m, {"a": 3.0})   # b 欠損
    assert c["b"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_explain.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'keiba.explain'`

- [ ] **Step 3: Write minimal implementation**

`keiba/explain.py` を新規作成:

```python
"""学習済みロジスティック回帰モデルの、馬ごとの予想に対する特徴量の寄与を
分解する(線形アトリビューション)。表示している勝率と同じモデルから根拠を出す
ので、資料として矛盾しない。

寄与 = 係数_i × 標準化した特徴量_i (= 対数オッズへの押し上げ/押し下げ)。
"""

# 市場(オッズ)由来の特徴量。根拠表示では除外する(=オッズに無い要因だけ見せる)。
MARKET_COLUMNS = {"win_odds", "log_odds", "odds_rank_pct", "popularity",
                  "prev_avg_log_odds", "prev_avg_popularity"}

# 特徴量 -> 表示用の日本語ラベル。
FEATURE_LABELS_JA = {
    "win_odds": "単勝オッズ", "popularity": "人気", "log_odds": "オッズ(対数)",
    "odds_rank_pct": "オッズ順位", "prev_avg_log_odds": "近走の人気度",
    "prev_avg_popularity": "近走の人気",
    "age": "年齢", "weight_carried": "斤量", "body_weight": "馬体重",
    "distance": "距離", "surface_turf": "芝/ダート",
    "track_condition_score": "馬場状態", "field_size": "頭数",
    "body_weight_z": "馬体重(相対)", "weight_carried_z": "斤量(相対)",
    "prev_runs": "出走数", "prev_win_rate": "近走勝率",
    "prev_avg_finish": "近走着順", "prev_avg_last3f": "近走上がり",
}


def feature_contributions(model, feature_row) -> dict:
    """{特徴量: 対数オッズへの寄与} を返す。

    model は Pipeline(StandardScaler, LogisticRegression) を想定し、
    model.feature_columns_ の順で寄与 = coef_i * (x_i - mean_i)/scale_i。
    欠損/数値化不能/NaN の特徴量は 0 寄与。
    """
    cols = list(getattr(model, "feature_columns_", []))
    scaler = model.named_steps["scaler"]
    clf = model.named_steps["clf"]
    coef = clf.coef_[0]
    out = {}
    for i, c in enumerate(cols):
        v = feature_row.get(c) if hasattr(feature_row, "get") else None
        try:
            v = float(v)
        except (TypeError, ValueError):
            out[c] = 0.0
            continue
        if v != v:  # NaN
            out[c] = 0.0
            continue
        z = (v - scaler.mean_[i]) / scaler.scale_[i]
        out[c] = float(coef[i] * z)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_explain.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add keiba/explain.py tests/test_explain.py
git commit -m "feat: explain.feature_contributions (linear attribution)"
```

---

### Task 4: `deviation_reasons` (非オッズ要因 top_n)

**Files:**
- Modify: `keiba/explain.py`
- Test: `tests/test_explain.py`

- [ ] **Step 1: Write the failing test**

`tests/test_explain.py` に追記:

```python
from keiba.explain import deviation_reasons


def test_deviation_reasons_excludes_market_and_sorts_by_abs():
    contribs = {"win_odds": 5.0, "age": -0.4,
                "body_weight": 0.2, "prev_avg_finish": -0.9}
    r = deviation_reasons(contribs, top_n=2)
    feats = [f for f, _ in r]
    assert "win_odds" not in feats          # 市場要因は除外
    assert r[0][0] == "prev_avg_finish"     # |−0.9| が最大
    assert len(r) == 2


def test_deviation_reasons_drops_zero():
    assert deviation_reasons({"age": 0.0, "body_weight": 0.0}) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_explain.py -q`
Expected: FAIL — `ImportError: cannot import name 'deviation_reasons'`

- [ ] **Step 3: Write minimal implementation**

`keiba/explain.py` に追加:

```python
def deviation_reasons(contributions: dict, exclude=MARKET_COLUMNS, top_n=3):
    """オッズ系を除外し、|寄与| の大きい順に最大 top_n を返す。

    返り値: [(特徴量名, 寄与値), ...]。寄与が 0 の要因は含めない。
    """
    items = [(f, c) for f, c in contributions.items()
             if f not in exclude and c != 0.0]
    items.sort(key=lambda kv: abs(kv[1]), reverse=True)
    return items[:top_n]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_explain.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add keiba/explain.py tests/test_explain.py
git commit -m "feat: explain.deviation_reasons (top non-odds factors)"
```

---

### Task 5: `attach_reasons` (乖離馬の予想行へ根拠を付与)

**Files:**
- Modify: `keiba/explain.py`
- Test: `tests/test_explain.py`

- [ ] **Step 1: Write the failing test**

`tests/test_explain.py` に追記(`_model` を再利用):

```python
import pandas as pd
from keiba.explain import attach_reasons


def test_attach_reasons_only_for_deviating_horses():
    m = _model()
    preds = [{"name": "X", "score": 2}, {"name": "Y", "score": 0}]
    feats = pd.DataFrame({"name": ["X", "Y"], "a": [3.0, 1.0], "b": [2.0, 4.0]})
    attach_reasons(preds, m, feats)
    assert "reasons" in preds[0] and len(preds[0]["reasons"]) >= 1
    # 各要因は (日本語ラベル, '↑' or '↓') の形
    label, arrow = preds[0]["reasons"][0]
    assert isinstance(label, str) and arrow in ("↑", "↓")
    assert "reasons" not in preds[1]          # |score|<1 は付けない


def test_attach_reasons_skips_missing_feature_row():
    m = _model()
    preds = [{"name": "Z", "score": 3}]        # feats に Z が無い
    feats = pd.DataFrame({"name": ["X"], "a": [3.0], "b": [2.0]})
    attach_reasons(preds, m, feats)
    assert "reasons" not in preds[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_explain.py -q`
Expected: FAIL — `ImportError: cannot import name 'attach_reasons'`

- [ ] **Step 3: Write minimal implementation**

`keiba/explain.py` に追加:

```python
def attach_reasons(predictions, model, features, min_abs_score=1, top_n=3):
    """|score| >= min_abs_score の予想行へ、非オッズ根拠を `reasons` として付ける。

    predictions: predict_ranking が返す行のリスト(各行に 'name','score')。
    features: build_features の DataFrame(列に 'name' と特徴量)。
    reasons の各要素は (日本語ラベル, '↑'|'↓')。'↑'=勝率を押し上げた要因。
    特徴量行が見つからない馬はスキップ。
    """
    for row in predictions:
        if abs(row.get("score", 0)) < min_abs_score:
            continue
        match = features[features["name"] == row["name"]]
        if match.empty:
            continue
        feature_row = match.iloc[0].to_dict()
        contribs = feature_contributions(model, feature_row)
        reasons = deviation_reasons(contribs, top_n=top_n)
        row["reasons"] = [(FEATURE_LABELS_JA.get(f, f),
                           "↑" if c > 0 else "↓") for f, c in reasons]
    return predictions
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_explain.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add keiba/explain.py tests/test_explain.py
git commit -m "feat: explain.attach_reasons (reasons on deviating horses)"
```

---

### Task 6: `assemble_report`/`build_race_report` に model+features を配線

**Files:**
- Modify: `keiba/report.py` (`assemble_report`, `build_race_report`)
- Test: `tests/test_report.py`

- [ ] **Step 1: Write the failing test**

`tests/test_report.py` に追記(既存の `_horse` ヘルパを利用):

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keiba.features import build_features


def _two_feature_model():
    X = pd.DataFrame({"age": [3.0, 4, 5, 6], "body_weight": [460.0, 480, 500, 520]})
    y = [0, 0, 1, 1]
    m = Pipeline([("scaler", StandardScaler()),
                  ("clf", LogisticRegression(max_iter=1000))])
    m.fit(X, y)
    m.feature_columns_ = ["age", "body_weight"]
    return m


def test_assemble_report_attaches_reasons_for_deviating():
    horses = [_horse("A", 1, 2.0), _horse("B", 2, 5.0), _horse("C", 3, 8.0),
              _horse("D", 4, 12.0), _horse("E", 5, 20.0)]
    race = Race(race_id="r1", name="テストS", date="d", course="東京",
                distance=1600, surface="芝", turn="右", track_condition="良",
                weather="晴", horses=horses)
    win = {h.name: p for h, p in zip(horses, [0.4, 0.25, 0.15, 0.12, 0.08])}
    odds = {"win": {}, "place": {}, "quinella": {}, "wide": {}}
    feats = build_features(race)
    model = _two_feature_model()
    rep = assemble_report(race, win, odds, model=model, features=feats)
    deviating = [p for p in rep["predictions"] if abs(p["score"]) >= 1]
    assert deviating, "テストデータに乖離馬が居ること"
    assert any("reasons" in p for p in deviating)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_report.py::test_assemble_report_attaches_reasons_for_deviating -q`
Expected: FAIL — `TypeError: assemble_report() got an unexpected keyword argument 'model'`

- [ ] **Step 3: Write minimal implementation**

`keiba/report.py` の `assemble_report` を変更:

```python
def assemble_report(race, win_probs: dict, odds: dict,
                    model=None, features=None) -> dict:
    """予想ランキングと買い目を、表示しやすい dict にまとめる。

    model と features(build_features の DataFrame)が与えられれば、乖離馬に
    非オッズ根拠 reasons を付ける。
    """
    predictions = predict_ranking(race, win_probs)
    if model is not None and features is not None:
        from keiba.explain import attach_reasons
        attach_reasons(predictions, model, features)
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
```

`keiba/report.py` の `build_race_report` 末尾を変更(モデルと特徴量を渡す):

```python
    df = build_features(race)
    model = (load_model(DEFAULT_MODEL_PATH)
             if os.path.exists(DEFAULT_MODEL_PATH) else None)
    win_probs = win_probabilities(df, model=model)
    return assemble_report(race, win_probs, odds, model=model, features=df)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_report.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add keiba/report.py tests/test_report.py
git commit -m "feat: wire model+features into assemble_report for reasons"
```

---

### Task 7: `main.py` を予想ダッシュボード表示に刷新

**Files:**
- Modify: `main.py` (全面置換)

- [ ] **Step 1: 新しい `main.py` を書く**

`main.py` を以下で全置換:

```python
import sys
from keiba.report import build_race_report


def _fmt_score(s: int) -> str:
    return f"+{s}" if s > 0 else (str(s) if s < 0 else "0")


def main(race_id: str, enrich: bool = False):
    print(f"レース {race_id} を取得・計算中...")
    if enrich:
        print("  各馬の過去成績も取得します(時間がかかります)...")
    rep = build_race_report(race_id, enrich=enrich)
    m = rep["meta"]
    print(f"  {m['name']} / {m['surface']}{m['distance']}m / {m['field_size']}頭")

    print("\n=== ① 予想ダッシュボード(資料) ===")
    print("  予想=モデル勝率 / 市場=オッズが示す勝率")
    print("  評価=市場との乖離 -5〜+5(+は過小評価/妙味・-は過大評価)")
    print(f"  {'評価':>4} {'馬名':<11} {'予想':>6} {'市場':>6} {'複勝':>5} 確信")
    for r in rep["predictions"]:
        pp = f"{r['place_prob']:.0%}" if r["place_prob"] is not None else "—"
        print(f"  {_fmt_score(r['score']):>4} {r['name']:<11} "
              f"{r['win_prob']:>6.1%} {r['market_prob']:>6.1%} {pp:>5} {r['level']}")

    print("\n=== ② 乖離している馬の根拠(オッズに無い要因) ===")
    devs = [r for r in rep["predictions"] if r.get("reasons")]
    if not devs:
        print("  市場と概ね一致。際立った乖離馬はありません。")
    for r in devs:
        factors = " ・ ".join(f"{lab}{arrow}" for lab, arrow in r["reasons"])
        print(f"  {r['name']}(評価{_fmt_score(r['score'])} "
              f"予想{r['win_prob']:.0%}): {factors}")

    print("\n=== ③ 参考: 買い目のEV ===")
    print("  ※ 市場効率につき控除率を越える優位性は確認されていません(参考値)。")
    if not rep["bets"]:
        print("  該当なし。")
    else:
        for i, b in enumerate(rep["bets"], 1):
            mark = "＋" if b["ev"] >= 0 else "－"
            print(f"  {i}. [{b['type']}] {b['sel']:<11} オッズ{b['odds']:>6} "
                  f"推定{b['prob']:5.1%} EV{b['ev']:+.2f}{mark}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    enrich = "--enrich" in sys.argv
    if not args:
        print("使い方: python main.py <race_id> [--enrich]")
        sys.exit(1)
    main(args[0], enrich=enrich)
```

- [ ] **Step 2: 構文チェック(インポートが通ること)**

Run: `python -c "import main"`
Expected: エラーなく終了(出力なし)。

- [ ] **Step 3: Commit**

```bash
git add main.py
git commit -m "feat: main.py prediction dashboard (score/predicted%/reasons)"
```

---

### Task 8: 全テスト確認と手動スモーク

**Files:** なし(検証のみ)

- [ ] **Step 1: 全スイート実行**

Run: `python -m pytest -q`
Expected: 全テスト PASS(既存 + 新規)。

- [ ] **Step 2: 手動スモーク(ネットワーク・任意)**

Run: `python main.py 202602010101 --enrich`
Expected: ①全馬一覧(評価/予想/市場/複勝/確信) → ②乖離馬の根拠 → ③参考EV、が表示され、例外で落ちない。`model.pkl` が無い環境では②が「際立った乖離馬はありません」または市場ベース表示になっても可(クラッシュしないこと)。

- [ ] **Step 3: ブランチ完了の確認**

Run: `git log --oneline -8`
Expected: Task1〜7 のコミットが並ぶ。Phase 1 完了。
```
