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


def deviation_reasons(contributions: dict, exclude=MARKET_COLUMNS, top_n=3):
    """オッズ系を除外し、|寄与| の大きい順に最大 top_n を返す。

    返り値: [(特徴量名, 寄与値), ...]。寄与が 0 の要因は含めない。
    """
    items = [(f, c) for f, c in contributions.items()
             if f not in exclude and c != 0.0]
    items.sort(key=lambda kv: abs(kv[1]), reverse=True)
    return items[:top_n]


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
