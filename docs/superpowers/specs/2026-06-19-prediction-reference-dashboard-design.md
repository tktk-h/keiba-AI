# 予想の資料ダッシュボード(CLI) — 設計

日付: 2026-06-19

## 背景と目標
一連の検証で「自動で市場に勝つ(回収率>100%)」は現データでは不可能と確定した
(市場はほぼ完璧にキャリブレート、モデル/特徴量/買い方を変えても控除率の壁~80%)。
そこで目標を**転換**する:

> ツールは利益機械ではなく、**人間(ユーザー)の予想を補助する「資料」**になる。

ユーザーが選んだ価値: ①市場との乖離ハイライト ②各馬の一覧ダッシュボード
③予想根拠の提示(乖離馬のみ)。提供形態は **CLI (main.py) の出力強化**。

## スコープ
- `main.py` の出力を「予想の資料」ダッシュボードに刷新。
- 新規 `keiba/explain.py`: ロジスティック回帰の線形寄与アトリビューション(純粋・TDD)。
- `build_race_report` で乖離馬に `reasons` を付与。
- 予想ロジック(`recommend.py`)・モデル・Web UI・データ収集は変更しない。

## 非目標 (YAGNI)
- モデルや特徴量の変更、新データ取得。
- 買い目で優位性を出すこと(出ないと確定済み)。
- Web UI の変更。
- LGBM 等の非線形モデルの寄与(SHAP)。現行の本番モデルはロジスティック回帰なので
  線形寄与で十分。将来モデルを変えたら別途。

## 出力デザイン (main.py)

### ① 一覧ダッシュボード(全馬)
推定勝率の高い順に1行ずつ:
`印 / 馬名 / モデル勝率 / 複勝率 / 市場勝率 / 乖離 / 確信度`
- 乖離列: 妙味=◎、過剰人気=▲、互角=・ (既存 `_value_tag` を流用)。
- 乖離馬は印で視覚強調。

### ② 乖離馬の根拠(新規)
①で妙味/過剰人気と判定された馬についてのみ、評価を動かした
**非オッズ要因 上位2〜3個**を表示。
```
馬A (妙味◎): 近走着順 ↑ ・ 馬体重 ↓
馬B (過剰人気▲): 出走間隔 ↓ ・ 年齢 ↓
```
- `↑` = その要因が勝率を押し上げた / `↓` = 押し下げた。
- 値はオッズに無い要因に限る(オッズ系6特徴量を除外)。

### ③ 買い目セクションの格下げ
現状の「② 買い目のEV(推奨)」を、優位性なしと明示する**参考扱い**に変更:
> 参考: +EV候補(市場効率につき控除率を越える優位性は確認されていません)

誤って「推奨」と読まれないよう文言で明確化。買い目の計算自体は現状維持。

## 新規モジュール `keiba/explain.py` (純粋関数)

```
MARKET_COLUMNS = {"win_odds","log_odds","odds_rank_pct","popularity",
                  "prev_avg_log_odds","prev_avg_popularity"}

feature_contributions(model, feature_row) -> dict[str, float]
    # model: Pipeline(StandardScaler, LogisticRegression)。
    # 各特徴量の対数オッズへの寄与 = coef_i * (x_i - mean_i)/scale_i。
    # feature_row: model.feature_columns_ をキーに持つ dict/Series。
    # 欠損/数値化不能はその特徴量を 0 寄与として扱う。

deviation_reasons(contributions, exclude=MARKET_COLUMNS, top_n=3)
    -> list[tuple[str, float]]
    # exclude を除外し、|寄与| の大きい順に最大 top_n、寄与が 0 の要因は出さない。

FEATURE_LABELS_JA: dict[str, str]   # 特徴量 -> 日本語ラベル(表示用)
```

設計理由: 表示している勝率(モデル)と根拠が**必ず同じモデルから**出るので、
資料として矛盾しない。線形モデルなので寄与は厳密に分解できる。

## 配線
- `build_race_report` は既に `df`(=build_features) と `model` を持つ。
  `assemble_report` 後、`predictions` のうち乖離タグの行に対して
  `df` の該当馬の特徴量行から `feature_contributions` → `deviation_reasons` を計算し、
  各 prediction 行へ `reasons: list[(label_ja, direction)]` を付与する。
- `assemble_report` は純粋関数のまま(根拠付与は `build_race_report` 側、または
  model/features を受け取れるよう引数追加。実装時に最小変更で選択)。
- `main.py` は新フォーマットで ①②③ を表示。

## エラー処理
- `model` が無い(model.pkl 不在)場合: 根拠セクションは出さず、①は市場ベース勝率で表示。
- 乖離馬が 0 頭: 「乖離している馬はありません(市場と概ね一致)」と表示。
- 特徴量行が見つからない/欠損: その馬の根拠はスキップ(クラッシュしない)。

## テスト (TDD)
- `keiba/explain.py`:
  - `feature_contributions` が coef×標準化値を返す(既知の係数/スケーラで検算)。
  - 欠損特徴量は 0 寄与。
  - `deviation_reasons` が除外集合を外し、|寄与|順 top_n、0寄与を除く。
  - ラベル表に主要特徴量が存在。
- 既存の `report`/`recommend` のテストを壊さない(全スイート緑)。
- main.py の表示は薄い結合層なので手動確認 + 既存レポートテストで担保。

## 成功基準
- `python main.py <race_id> --enrich` で、全馬一覧 + 乖離馬の非オッズ根拠 +
  参考扱いの買い目、が表示される。
- 表示勝率と根拠が同一モデル由来で矛盾しない。
- 全テスト緑。
