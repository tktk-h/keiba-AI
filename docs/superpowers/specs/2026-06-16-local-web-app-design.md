# ローカル競馬予想Webアプリ 設計

## 目的
race_idを手で打たず、**今日の開催レースを一覧から選ぶだけ**で、
①予想(モデル vs 市場・妙味)と②EV参考をブラウザで見られるローカルWebアプリ。
このPCで完結(ローカル起動)。将来スマホ公開できるよう疎結合に保つ。

## 背景
予想ロジック(出馬表→過去成績→モデル→オッズ→予想/EV)は CLI `main.py` で
完成済み。本アプリはその上に「レース選択UI + ブラウザ表示」を載せる。利益は
出せないと実証済み(`docs/analysis/`)なので、②EVは「参考・優位性未確認」と
明示する既存方針を踏襲する。

## 使い方(完成形)
1. `python run_web.py` → `http://localhost:5000` をブラウザで開く
2. 今日の開催レース一覧(会場・R番号・レース名)を表示 → クリック
3. そのレースの ①予想 + ②EV参考 を表示

## コンポーネント

### 1. `keiba/report.py`(新規)— CLI/Web共通の組み立て
- `assemble_report(race, win_probs, odds) -> dict`(純粋・テスト対象):
  `predict_ranking` と `recommend_all` を呼び、表示に必要な dict を返す:
  ```
  {
    "meta": {"race_id","name","surface","distance","field_size"},
    "predictions": [ {name,win_prob,market_prob,place_prob,value,level,confidence}, ... ],
    "bets": [ {type,sel,prob,odds,ev,level}, ... ],
    "any_positive": bool,
  }
  ```
- `build_race_report(race_id, enrich=True) -> dict`(ネットワーク):
  `scrape_race` → 確定オッズで単勝/人気を補完 → `build_features` →
  `win_probabilities(model)` → `fetch_odds` → `assemble_report`。
  `main.py` の現行処理をこの関数へ移し、CLIもこれを呼ぶ(DRY)。

### 2. `keiba/race_list.py`(拡張)— 今日のレース一覧
- `fetch_race_cards(kaisai_date) -> list[dict]`:
  各レースを `{race_id, venue, number, name}` で返す。
  - `venue`/`number` は race_id から導出(信頼できる):
    race_id = YYYY + 場コード(2) + 開催(2) + 日(2) + R(2)。
    場コード表 {01札幌,02函館,03福島,04新潟,05東京,06中山,07中京,08京都,
    09阪神,10小倉}。
  - `name`(レース名)は一覧ページからパースできれば付ける。取れなければ空。
- `today_cards() -> list[dict]`: 今日(ローカル日付)の `fetch_race_cards`。

### 3. `web/app.py`(新規・Flask)
- `GET /` … `fetch_race_cards(date)` を会場ごとにグルーピングして一覧表示。
  `?date=YYYYMMDD` で日付上書き可(非開催日・検証用の保険)。開催が無ければ
  「本日の開催はありません」を表示。
- `GET /race/<race_id>` … `build_race_report`(キャッシュ)を呼んで①②を表示。
- メモリキャッシュ `{race_id: report}` で2回目以降は即時。

### 4. `web/templates/`(Jinja2)
- `index.html`: 会場ごとのレースカード一覧(リンク)。
- `race.html`: ①予想テーブル(予想%/市場%/妙味タグ/複勝%/確信度)+
  ②EV参考テーブル(参考ラベル+バックテスト注記)。
- スマホでも崩れない簡素なCSS、`<meta name="viewport">` 対応。

### 5. 起動・依存
- `run_web.py`: Flask アプリを起動(host=127.0.0.1, port=5000)。
- `requirements.txt` に `flask` を追加。

## データフロー
```
GET /            → fetch_race_cards(today) → index.html
GET /race/<id>   → build_race_report(id)[cache] → race.html
   build_race_report: scrape_race(enrich) → オッズ補完 → build_features
                      → win_probabilities(model) → fetch_odds → assemble_report
```

## UX上の割り切り(v1)
- 初回表示は過去成績取得で 30〜60秒。各馬の取得間隔を 0.5 秒へ短縮し、
  画面に「計算中は時間がかかります」を明示。キャッシュで2回目以降は即時。
- 同期処理(裏ジョブ・非同期は将来)。

## テスト(TDD)
- `assemble_report`(純粋):合成 Race + win_probs + odds を渡し、返り値の
  キー構成・predictions が勝率降順・bets が EV 降順・妙味タグの存在を検証。
- `fetch_race_cards` のパース:保存した一覧ページ風フィクスチャで
  race_id 抽出・venue/number 導出・name パースを検証。
- Flask ルート:test client で `/`(200・一覧描画)と、`build_race_report` を
  モックした `/race/<id>`(200・予想テーブル描画)をスモーク。
- 既存テストを壊さない。

## 進め方(リスク順)
1. `keiba/report.py`(assemble_report → build_race_report)+ テスト、`main.py` を委譲に
2. `race_list.fetch_race_cards` / `today_cards` + テスト
3. Flask アプリ + テンプレート + test client スモーク
4. `run_web.py` 起動・実レースで手動確認

## スコープ外(将来)
- スマホ公開(Cloudflare Tunnel 等)、非同期/バックグラウンド計算、
  日付カレンダーUI、ログイン、3連系・資金配分。

## 検証方法
- `pytest -q` 全通過(既存 + 新規)
- `python run_web.py` → `http://localhost:5000` で今日(または `?date=` 指定日)の
  一覧が出て、レースを選ぶと①②が表示されることを手動確認
