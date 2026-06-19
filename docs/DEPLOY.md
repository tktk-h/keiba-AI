# 公開手順（競馬場のスマホから使う）

アプリは Flask（`web/app.py` の `app`）。予想は `/race/<id>` がシェル(スピナー)を返し、
`/race/<id>/data` が JSON を返す。モデルは `model.pkl`（リポジトリ同梱）。

## 方法A: Render 常駐（PC不要・どこでも）

前提: Render アカウント、このリポジトリを GitHub(private可) に push 済み。

1. GitHub に push（リモートが無ければ作成して push）:
   ```
   gh repo create keiba-dashboard --private --source=. --remote=origin --push
   ```
2. Render ダッシュボード → New → Blueprint → この GitHub リポジトリを選択。
   `render.yaml` を自動検出してデプロイ（Python / gunicorn / `0.0.0.0:$PORT`）。
3. 発行された `https://keiba-dashboard.onrender.com` をスマホで開く。
   - 一覧は今日の開催。`?date=YYYYMMDD` で日付指定可。

注意:
- 無料プランは 15 分アクセスが無いとスリープ。次回は起動 30〜60 秒 ＋ enrich 30 秒で
  初回 1〜1.5 分待ち（以後は温かく速い）。
- 公開URLを知る人は誰でもアクセス＝スクレイピングを発火できる。URL は共有しない。
- `model.pkl` は同梱済み（無いと予想が市場ベースに退化し、評価/根拠が出ない）。

## 方法B: cloudflared トンネル（PC起動中・最速で試せる・無料・アカウント不要）

1. PC で LAN 公開してアプリ起動:
   ```
   set KEIBA_HOST=0.0.0.0    # PowerShell: $env:KEIBA_HOST="0.0.0.0"
   python run_web.py
   ```
2. 別ウィンドウで一時トンネルを張る:
   ```
   cloudflared tunnel --url http://localhost:5000
   ```
   表示される `https://xxxx.trycloudflare.com` をスマホで開く（PC起動中のみ有効）。

## 方法C: 同じWi-Fiだけ（自宅で動作確認）

`KEIBA_HOST=0.0.0.0 python run_web.py` → スマホで `http://<PCのIP>:5000`。
（競馬場では使えない。まず手元のスマホ表示確認用。）
