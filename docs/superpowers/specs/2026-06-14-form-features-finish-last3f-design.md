# 設計: 過去成績特徴に finish/last_3f を追加

最終更新: 2026-06-14

## 目的
`keiba/form_features.py` の過去成績特徴(FORM_COLUMNS)に、過去レースの
着順(finish)と上がり3F(last_3f)を使った特徴を追加し、再収集データで
バックテスト改善を検証する。

## 変更点

### 1. keiba/form_features.py
FORM_COLUMNS に2列追加:
- `prev_avg_finish`: 過去レースの平均着順(既存の `_prior_mean` パターンを再利用)
- `prev_avg_last3f`: 過去レースの平均上がり3F秒
  - `last_3f` の欠損は列全体の平均で補完してから集計する(0埋めは「速い」側に
    大きく偏るため)
- 初出走馬(prev_runs==0)は既存パターンと同様 `fillna(0.0)`

### 2. keiba/dataset.py
- `build_dataset` で `add_form_features()` を呼び出す(現状どこからも
  呼ばれていない)
- `finish` / `last_3f` は form特徴の計算にのみ使い、最終データセットには
  残さない(現在のリーク回避方針を維持)
- `FEATURE_COLUMNS` 自体は変更しない(基本5+相対5=10のまま)。
  `FORM_COLUMNS`(既存4+新規2=6)はデータセットの列として出力されるが、
  デフォルトのモデル学習には使わない

## スコープ外
- 既存の FORM_COLUMNS をデフォルトの FEATURE_COLUMNS に組み込むことは
  今回は行わない。組み込むには `main.py` のライブ予測経路
  (`keiba/features.py` の `build_features`)でも同じ列を計算できる必要が
  あり、別工数。今回の検証結果を見て次のステップで判断する。

## テスト
- `tests/test_form_features.py`: prev_avg_finish / prev_avg_last3f の
  3run検証パターンを追加
- `tests/test_model.py` (build_dataset): 最終列に finish/last_3f が
  含まれないこと、FORM_COLUMNS が含まれることを確認

## 検証手順
1. 1ヶ月分を新スキーマで再収集
2. `run_backtest(df, feature_columns=FEATURE_COLUMNS + FORM_COLUMNS)` で
   旧版(4特徴)と新版(6特徴)を比較
3. 改善が見られれば2年分を再収集し、model.pkl 再学習(本番組み込みは別途判断)
