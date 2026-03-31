import json
from backend.ml_pipeline import NBAPredictor

predictor = NBAPredictor(models_dir="backend/models")

try:
    # OKC vs BKN
    t1 = 1610612760
    t2 = 1610612751

    # Let's inspect raw probabilities
    row_t1 = predictor._get_team_row(t1)
    row_t2 = predictor._get_team_row(t2)

    def make_features(team_row, is_home):
        row = {}
        for feat in predictor.clf_features:
            if feat in team_row.index:
                row[feat] = team_row[feat]
            else:
                row[feat] = 0.0
        if is_home and "expectedTeamScore" in row:
            row["expectedTeamScore"] = row.get("expectedTeamScore", 0) + 2.5
        import pandas as pd
        return pd.DataFrame([row])

    X1 = make_features(row_t1, True)
    X2 = make_features(row_t2, False)

    prob1_raw = float(predictor.clf.predict_proba(X1)[0][1])
    prob2_raw = float(predictor.clf.predict_proba(X2)[0][1])

    print(f"OKC Raw: {prob1_raw:.4f}")
    print(f"BKN Raw: {prob2_raw:.4f}")

    # Log5 formulation
    p1 = prob1_raw
    p2 = prob2_raw
    denom_log5 = p1 + p2 - 2 * p1 * p2
    if denom_log5 > 0:
        log5_1 = (p1 - p1 * p2) / denom_log5
    else:
        log5_1 = 0.5
    
    print(f"Log5 OKC: {log5_1:.4f}")
    print(f"Log5 BKN: {1 - log5_1:.4f}")

    print("Current API output:", json.dumps(predictor.predict(t1, t2), indent=2))

except Exception as e:
    import traceback
    traceback.print_exc()
