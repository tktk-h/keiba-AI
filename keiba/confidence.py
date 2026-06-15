"""推定値の信頼度スコア。

「市場に勝てる自信」ではなく、その確率/EV推定をどれだけ信用してよいかを表す。
診断(docs/superpowers/specs/2026-06-15-ev-recommendation-ai-design.md 参照)で
モデルは較正済みだが市場超えはしないと分かっている。この信頼度はデータ品質に
基づく値であり、利益を約束するものではない。
"""


def confidence(data_runs: int, sharpness: float,
               odds_confirmed: bool, field_ok: bool, hit_prob=None):
    """0–1 のスコアと 低/中/高 を返す。

    data_runs:  関与馬の過去走数(連系は最小値)。多いほど高い(5で頭打ち)。
    sharpness:  抜けた本命の明確さ 0–1(混戦ほど低い)。
    odds_confirmed: 対象オッズが確定済み(None でない)か。
    field_ok:   出走頭数が妥当(5–18)か。
    hit_prob:   買い目の当選確率(任意)。渡すと「当たりやすさ=EV推定の頑健さ」を
                加味する。極小確率の穴(0.5%等)は確率推定の誤差でEVが大きく振れる
                ため低くなる。予想(全馬ランキング)では渡さない=データ品質のみ。
    """
    data = min(max(data_runs, 0), 5) / 5.0
    sharp = min(max(sharpness, 0.0), 1.0)
    odds = 1.0 if odds_confirmed else 0.5
    field = 1.0 if field_ok else 0.7
    base = 0.4 * data + 0.3 * sharp + 0.15 * odds + 0.15 * field
    if hit_prob is None:
        score = base
    else:
        robust = min(max(hit_prob, 0.0) / 0.05, 1.0)   # 5%以上で満点
        score = 0.6 * base + 0.4 * robust
    score = round(score, 3)
    if score >= 0.66:
        level = "高"
    elif score >= 0.40:
        level = "中"
    else:
        level = "低"
    return score, level
