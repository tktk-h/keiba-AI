# 引き継ぎ資料(中央競馬 期待値予想プログラム)

最終更新: 2026-06-14

## プロジェクトの目的
JRA(中央競馬)のレースについて、出走馬の情報をもとに機械学習で各馬の勝率を予測し、
期待値の高い買い方を提案する。最終目標は「複数券種を比較して一番期待値が高い買い方を提案」。
作者の方針: できるだけ精度を高くしたい。安易に「勝てる」と言わず、必ずバックテストで検証する。

## 作業ディレクトリ
`C:\Users\0525t\OneDrive - 同志社大学\ポートフォリオ\claude\競馬`
git管理(mainブランチ)。Python 3.14、Windows、PowerShell。
注意: netkeibaはEUC-JPエンコーディング。コンソール出力は `PYTHONIOENCODING=utf-8` を付ける。

## 現状: どこまで完成したか
出馬表取得 → 過去成績・血統取得 → 機械学習で勝率予測 → 期待値計算 → 買い目推奨 →
レースID自動収集 → バックテスト、まで一通り完成。**全39テストパス。**

### モジュール構成(keiba/)
- `models.py` — Race / Horse / PastRun データクラス
- `scraper.py` — netkeiba出馬表ページ取得(`scrape_race(race_id, enrich=)`)
- `horse_page.py` — 各馬の過去成績・血統ページ取得
- `result_page.py` — レース結果ページ(着順=正解ラベル)取得
- `race_list.py` — カレンダー/日別一覧からレースID自動収集
- `dataset.py` — 学習用データセット構築。FEATURE_COLUMNS定義
- `relative_features.py` — レース内相対特徴(log_odds, odds_rank_pct, z-score等)
- `form_features.py` — 過去成績特徴(prev_runs, prev_win_rate等)を既存データから生成
- `predictor.py` — win_probabilities(model=)。place_probabilities(ハーヴィル法、未使用)
- `expected_value.py` — win_ev / place_ev / combo_prob
- `model.py` — train_model(ロジスティック回帰=採用) / train_lgbm_model(LightGBM=見送り)
- `recommend.py` — 単勝の期待値ランキング
- `backtest.py` — time_split(時系列分割) / run_backtest(回収率 vs 1番人気)

### スクリプト(リポジトリ直下)
- `main.py <race_id> [--enrich]` — 実レースを予想(model.pklがあれば使用)
- `train.py <race_id...> | --file | --month YYYY-MM [YYYY-MM]` — 収集して学習
- `backtest.py --csv data.csv | --month ...` — バックテスト

### 手元データ(.gitignore対象・再利用可)
- `data_combined.csv` — 2023+2024の2年分・6,910レース・約94,000頭分
- `data_months/*.csv` — 月別の生データ
- `model.pkl` — 全データで学習済みモデル

## 検証で分かったこと(重要)
1. **8ヶ月で出た回収率137%は幻**だった(少数サンプルの運)。2年に増やしたら100%前後に収束。
2. **過去成績特徴(form_features)は本物の改善**。賭け数が7〜10倍に増え、回収率が86〜106%に安定。
   1番人気ベースライン(約80%)と同等以上を1,000件超の賭けで安定して示す。
3. **LightGBMは効果なし**。賭けすぎ・過剰適合でロジスティック回帰に全分割で負け。採用見送り
   (コードは温存。特徴量が増えれば将来逆転の可能性)。
4. 競馬の控除率(約20%)の壁は厚く、1番人気の80%を安定して超えるのは難関。

### 最良モデル
ロジスティック回帰 + 全特徴(基本5 + 相対5 + 過去成績4 = 14特徴)。

## 既知の制約・注意点
- **券種は単勝のみ**。複勝・馬連・三連複は未実装(place_probabilities等の部品は一部あり)。
- バックテストは**確定オッズ**を使用。実際の購入は締切前(オッズ未確定)なので現実の回収率は下がり得る。
- データセットに着順(finish)・上がり(last_3f)・horse_idを保存していない(build_datasetが捨てている)。
  より細かい過去成績特徴を作るには収集パイプラインの小改修+再収集が1回必要。
- 大量収集は時間がかかる(1ヶ月約10分、2年で約2時間)。**スリープ無効化を必ず先に**
  (powercfg /change standby-timeout-ac 0 等)。過去に13時間ハングの事故あり。

## 次の一手(候補、優先順)
1. **複勝への拡張** — place_probabilities が既にある。当てやすくサンプルも増える自然な次の一歩。
2. **馬連・ワイド・三連複** — 当初設計の「複数券種を比較」の完成形。combo_probが土台。
3. **着順・上がりを保存する収集改修** → 前走着順・距離適性など細かい成績特徴で精度上乗せ。
4. 確定オッズ問題への対処(締切前オッズでの検証)。

## 開発スタイル
TDD(テスト先行)。各機能はkeiba/にモジュール分割+tests/に対応テスト。
作者は「おすすめは?」とよく聞く → 根拠を示して1つ推奨する。結果は誇張せず正直に報告。
