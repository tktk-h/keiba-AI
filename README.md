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
スクレイパーのセレクタは実際のnetkeibaページ構造に合わせて調整が必要です。
