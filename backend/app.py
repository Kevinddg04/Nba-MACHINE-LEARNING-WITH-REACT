"""
app.py
======
API Flask para el NBA ML Predictor.
Usa los modelos CatBoost entrenados en ml_pipeline.py.

SETUP:
    pip install flask flask-cors catboost scikit-learn joblib pandas
    python ml_pipeline.py      # entrenar modelos primero
    python app.py              # iniciar servidor
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from ml_pipeline import NBAPredictor, TEAM_NAMES
import random
import os
import subprocess

app = Flask(__name__)
CORS(app)

predictor = NBAPredictor()

CONFERENCE = {
    1610612738: "East", 1610612751: "East", 1610612752: "East",
    1610612753: "East", 1610612754: "East", 1610612741: "East",
    1610612739: "East", 1610612765: "East", 1610612749: "East",
    1610612748: "East", 1610612761: "East", 1610612755: "East",
    1610612764: "East", 1610612766: "East", 1610612763: "East",
    1610612743: "West", 1610612750: "West", 1610612740: "West",
    1610612757: "West", 1610612762: "West", 1610612744: "West",
    1610612745: "West", 1610612742: "West", 1610612746: "West",
    1610612747: "West", 1610612756: "West", 1610612759: "West",
    1610612760: "West", 1610612758: "West", 1610612737: "East",
}

def err(msg, code=400):
    return jsonify({"error": msg}), code


@app.route("/api/teams", methods=["GET"])
def get_teams():
    if not predictor.models_loaded:
        return err("Ejecuta ml_pipeline.py primero para entrenar los modelos.", 503)
    teams = predictor.get_all_teams()
    result = []
    for t in teams:
        tid = t["team_id"]
        # Buscar en snapshot las estadísticas del equipo
        row_df = predictor.snapshot[predictor.snapshot["teamId"] == tid]
        if row_df.empty:
            continue
        row = row_df.iloc[0]
        result.append({
            "team_id":               tid,
            "name":                  t["team_name"],
            "conf":                  CONFERENCE.get(tid, "Unknown"),
            "expectedScore":         round(float(row.get("expectedTeamScore", 0)), 1),
            "expectedOpponentScore": round(float(row.get("expectedOpponentScore", 0)), 1),
            "net_expected":          round(float(row.get("expectedTeamScore", 0)) - float(row.get("expectedOpponentScore", 0)), 1),
            "win_streak_5":          round(float(row.get("win_streak_5", 0)), 1),
            "RealHandicap":          round(float(row.get("RealHandicap", 0)), 1),
            "fg_pct":                round(float(row.get("fieldGoalsPercentage", 0)) * 100, 1),
        })
    return jsonify(result)


@app.route("/api/team/<int:team_id>", methods=["GET"])
def get_team(team_id):
    if not predictor.models_loaded:
        return err("Modelos no cargados.", 503)
    try:
        s = predictor.get_team_stats(team_id)
        s["conf"] = CONFERENCE.get(team_id, "Unknown")
        return jsonify(s)
    except ValueError as e:
        return err(str(e), 404)


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Body: { "team1": 1610612747, "team2": 1610612738, "home_team": "team1" }
    """
    if not predictor.models_loaded:
        return err("Modelos no cargados. Ejecuta ml_pipeline.py primero.", 503)

    data  = request.get_json()
    team1 = data.get("team1")
    team2 = data.get("team2")
    home  = data.get("home_team", "team1")

    if not team1 or not team2:
        return err("Campos requeridos: team1, team2")
    if team1 == team2:
        return err("Los equipos deben ser diferentes")
    if home not in ("team1", "team2", "neutral"):
        return err("home_team debe ser 'team1', 'team2' o 'neutral'")

    try:
        return jsonify(predictor.predict(int(team1), int(team2), home_team=home))
    except ValueError as e:
        return err(str(e), 404)
    except RuntimeError as e:
        return err(str(e), 503)


@app.route("/api/standings", methods=["GET"])
def standings():
    if not predictor.models_loaded:
        return err("Modelos no cargados.", 503)

    conf_filter = request.args.get("conf", "all")
    result = []
    for _, row in predictor.snapshot.iterrows():
        tid  = int(row["teamId"])
        conf = CONFERENCE.get(tid, "Unknown")
        if conf_filter != "all" and conf.lower() != conf_filter.lower():
            continue
        exp_score = float(row.get("expectedTeamScore", 0))
        exp_opp   = float(row.get("expectedOpponentScore", 0))
        result.append({
            "team_id":   tid,
            "name":      TEAM_NAMES.get(tid, f"Team {tid}"),
            "conf":      conf,
            "expectedScore":        round(exp_score, 1),
            "expectedOpponentScore":round(exp_opp, 1),
            "net_expected":         round(exp_score - exp_opp, 1),
            "win_streak_5":         round(float(row.get("win_streak_5", 0)), 1),
            "RealHandicap":         round(float(row.get("RealHandicap", 0)), 1),
            "fg_pct": round(float(row.get("fieldGoalsPercentage", 0)) * 100, 1),
        })
    result.sort(key=lambda x: x["net_expected"], reverse=True)
    return jsonify(result)


@app.route("/api/head-to-head", methods=["POST"])
def head_to_head():
    if not predictor.models_loaded:
        return err("Modelos no cargados.", 503)
    data  = request.get_json()
    t1_id = data.get("team1")
    t2_id = data.get("team2")
    if not t1_id or not t2_id:
        return err("Campos requeridos: team1, team2")
    try:
        t1 = predictor.get_team_stats(int(t1_id))
        t2 = predictor.get_team_stats(int(t2_id))
    except ValueError as e:
        return err(str(e), 404)

    random.seed(hash(str(t1_id) + str(t2_id)))
    games = []
    t1_wins = 0
    for i in range(5):
        s1 = round(t1["expectedTeamScore"] + random.uniform(-12, 12))
        s2 = round(t2["expectedTeamScore"] + random.uniform(-12, 12))
        winner_id = t1_id if s1 > s2 else t2_id
        if winner_id == t1_id:
            t1_wins += 1
        games.append({"game": i+1, "t1_score": s1, "t2_score": s2, "winner_id": winner_id})

    return jsonify({
        "team1_id": t1_id, "team2_id": t2_id,
        "team1_name": t1["team_name"], "team2_name": t2["team_name"],
        "team1_wins": t1_wins, "team2_wins": 5 - t1_wins,
        "games": games,
    })


@app.route("/api/model/info", methods=["GET"])
def model_info():
    if not predictor.models_loaded:
        return err("Modelos no cargados.", 503)
    clf = predictor.clf
    feat_imp = clf.get_feature_importance()
    top_feats = sorted(
        [{"feature": f, "importance": round(float(i), 2)}
         for f, i in zip(predictor.clf_features, feat_imp)],
        key=lambda x: -x["importance"]
    )
    return jsonify({
        "model_type":        "CatBoostClassifier",
        "iterations":        clf.tree_count_,
        "num_features":      len(predictor.clf_features),
        "top_10_features":   top_feats[:10],
        "teams_in_snapshot": len(predictor.snapshot),
    })

@app.route("/api/admin/update", methods=["GET", "POST"])
def admin_update():
    """
    Endpoint para ejecutar kaggle_fetcher y ml_pipeline.
    Protegido por variable de entorno ADMIN_SECRET.
    """
    secret = request.args.get("secret")
    env_secret = os.environ.get("ADMIN_SECRET", "supersecreto")

    if secret != env_secret:
        return err("No autorizado. Proveer secret via query_string.", 401)

    try:
        # CUIDADO: subprocess es bloqueante. Para la nube, podrías querer un thread o Redis Queue, 
        # pero es útil para cargas que toman pocos segundos/minutos.
        print("[Update] Iniciando descarga desde Kaggle...")
        subprocess.run(["python", "kaggle_fetcher.py", "--force"], check=True)
        
        print("[Update] Iniciando entrenamiento de modelos...")
        subprocess.run(["python", "ml_pipeline.py"], check=True)

        print("[Update] Recargando modelos en memoria...")
        predictor._load_models()

        return jsonify({"message": "Pipeline de actualización completado exitosamente."})
    except subprocess.CalledProcessError as e:
        return err(f"El comando falló con codigo {e.returncode}.", 500)
    except Exception as e:
        return err(f"Error inesperado: {str(e)}", 500)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
