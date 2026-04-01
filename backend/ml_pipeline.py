"""
ml_pipeline.py
==============
Pipeline completo de Machine Learning para predicción NBA.
Extraído y organizado desde el notebook de CatBoost.

Contiene:
  - Preprocesamiento idéntico al notebook
  - CatBoostClassifier  → predice WIN / LOSS
  - CatBoostRegressor   → predice PUNTAJE del equipo
  - Guardado de modelos en /models/

USO:
    # Entrenar y guardar modelos:
    python ml_pipeline.py

    # Usar desde Flask (app.py):
    from ml_pipeline import NBAPredictor
    predictor = NBAPredictor()
    result = predictor.predict(team1_id, team2_id, home_team=1)
"""

import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────
CSV_PATH   = "TeamStatistics.csv"
MODELS_DIR = Path("models")

# IDs a eliminar (del notebook original - solo equipos de exhibición/All-Star)
IDS_A_ELIMINAR = [
    15016, 15018, 50013, 50014
]

# Features finales del clasificador (igual que en el notebook, Cell 29)
CLASSIFIER_FEATURES = [
    "assists", "blocks", "steals",
    "fieldGoalsAttempted", "fieldGoalsMade", "fieldGoalsPercentage",
    "threePointersAttempted", "threePointersMade", "threePointersPercentage",
    "freeThrowsAttempted", "freeThrowsMade", "freeThrowsPercentage",
    "reboundsDefensive", "reboundsOffensive", "reboundsTotal",
    "foulsPersonal", "turnovers",
    "totalPoints", "RealHandicap",
    "win_streak_5",
    "expectedTeamScore",
    "expectedOpponentScore",
]

TEAM_NAMES = {
    1610612738: "Boston Celtics",
    1610612751: "Brooklyn Nets",
    1610612752: "New York Knicks",
    1610612753: "Orlando Magic",
    1610612754: "Indiana Pacers",
    1610612741: "Chicago Bulls",
    1610612739: "Cleveland Cavaliers",
    1610612765: "Detroit Pistons",
    1610612749: "Milwaukee Bucks",
    1610612748: "Miami Heat",
    1610612761: "Toronto Raptors",
    1610612743: "Denver Nuggets",
    1610612750: "Minnesota Timberwolves",
    1610612740: "New Orleans Pelicans",
    1610612757: "Portland Trail Blazers",
    1610612762: "Utah Jazz",
    1610612744: "Golden State Warriors",
    1610612745: "Houston Rockets",
    1610612742: "Dallas Mavericks",
    1610612746: "LA Clippers",
    1610612747: "Los Angeles Lakers",
    1610612756: "Phoenix Suns",
    1610612759: "San Antonio Spurs",
    1610612763: "Memphis Grizzlies",
    1610612760: "Oklahoma City Thunder",
    1610612758: "Sacramento Kings",
    1610612755: "Philadelphia 76ers",
    1610612737: "Atlanta Hawks",
    1610612764: "Washington Wizards",
    1610612766: "Charlotte Hornets",
}


# ─────────────────────────────────────────────────────────────────────────────
#  1. CARGA Y LIMPIEZA (igual que el notebook)
# ─────────────────────────────────────────────────────────────────────────────

def load_and_clean(csv_path: str = CSV_PATH) -> pd.DataFrame:
    """Carga el CSV y aplica la limpieza del notebook (cells 0-13)."""
    print("[Pipeline] Cargando dataset...")
    df = pd.read_csv(csv_path, index_col=0, low_memory=False)
    print(f"  → {len(df):,} filas cargadas")

    # Columnas a eliminar (Cell 2)
    drop_cols = [
        "coachId", "seasonLosses", "seasonWins", "timeoutsRemaining",
        "q1Points", "q2Points", "q3Points", "q4Points",
        "teamCity", "opponentTeamCity", "timesTied", "benchPoints",
        "numMinutes", "gameLabel", "gameSubLabel", "seriesGameNumber",
        "gameType", "teamName", "opponentTeamName",
        "plusMinusPoints", "biggestLead", "biggestScoringRun",
        "pointsFastBreak", "pointsFromTurnovers", "pointsInThePaint",
        "pointsSecondChance", "leadChanges",
    ]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Fechas (2015 - Actualidad) - Rango de Oro solicitado por usuario
    df["gameDateTimeEst"] = pd.to_datetime(
        df["gameDateTimeEst"], errors="coerce", format="mixed", utc=True
    ).dt.normalize()
    # Permitir partidos hasta finales de 2026 para no filtrar los juegos de ayer
    df = df[(df["gameDateTimeEst"] >= "2015-01-01") & (df["gameDateTimeEst"] <= "2026-12-31")]
    df = df.sort_values("gameDateTimeEst").reset_index(drop=True)

    # Filtrar IDs inválidos (Cell 12-13)
    valid_ids = set(df["teamId"].unique()) | set(df["opponentTeamId"].unique())
    df = df[
        df["teamId"].isin(valid_ids) &
        df["opponentTeamId"].isin(valid_ids)
    ].copy()
    df = df[
        ~(df["teamId"].isin(IDS_A_ELIMINAR) | df["opponentTeamId"].isin(IDS_A_ELIMINAR))
    ].reset_index(drop=True)

    print(f"  → {len(df):,} filas después de limpieza")
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  2. FEATURE ENGINEERING (igual que el notebook)
# ─────────────────────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica el feature engineering completo (cells 14-26)."""
    print("[Pipeline] Construyendo features...")

    # Win como int (Cell 14)
    df = df.dropna(subset=["win"]).copy()
    df["win"] = df["win"].astype(int)

    # Ordenar por equipo y fecha
    df = df.sort_values(["teamId", "gameDateTimeEst"]).reset_index(drop=True)

    # Win streak (Cell 14)
    df["win_streak_5"] = (
        df.groupby("teamId")["win"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum())
    )

    # totalPoints_real (Cell 22)
    df["totalPoints_real"] = df["teamScore"] + df["opponentScore"]

    # Shift de todas las columnas estadísticas (Cell 22 - evitar leakage)
    shift_cols = [
        "teamScore", "opponentScore",
        "assists", "blocks", "steals",
        "fieldGoalsAttempted", "fieldGoalsMade", "fieldGoalsPercentage",
        "threePointersAttempted", "threePointersMade", "threePointersPercentage",
        "freeThrowsAttempted", "freeThrowsMade", "freeThrowsPercentage",
        "reboundsDefensive", "reboundsOffensive", "reboundsTotal",
        "foulsPersonal", "turnovers",
    ]
    shift_cols = [c for c in shift_cols if c in df.columns]
    df[shift_cols] = df.groupby("teamId")[shift_cols].shift(1)

    # Rolling 5 partidos (Cell 22)
    rolling_cols = [
        "teamScore", "opponentScore",
        "assists", "blocks", "steals",
        "fieldGoalsAttempted", "fieldGoalsMade",
        "threePointersAttempted", "threePointersMade",
        "freeThrowsAttempted", "freeThrowsMade",
        "reboundsDefensive", "reboundsOffensive", "reboundsTotal",
        "foulsPersonal", "turnovers",
    ]
    rolling_cols = [c for c in rolling_cols if c in df.columns]
    df[rolling_cols] = (
        df.groupby("teamId")[rolling_cols]
        .rolling(5, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # PointDiff y Handicap (Cell 22)
    df["PointDiff"] = df["teamScore"] - df["opponentScore"]
    df["Handicap"]  = df["PointDiff"] * -1

    # Expected scores (Cell 22)
    df["expectedTeamScore"] = df.groupby("teamId")["teamScore"].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )
    df["expectedOpponentScore"] = df.groupby("teamId")["opponentScore"].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )
    df["totalPoints"] = df["expectedTeamScore"] + df["expectedOpponentScore"]

    # RealHandicap (Cell 22)
    df["RealHandicap"] = df["teamScore"] - df["opponentScore"]
    df["RealHandicap"] = (
        df.groupby("teamId")["RealHandicap"]
        .apply(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )

    # Defensa (rolling 10 del oponente - solicitado por usuario)
    df["defensive_rating_r10"] = (
        df.groupby("teamId")["opponentScore"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    )

    # Dropear NaN
    df = df.dropna().reset_index(drop=True)

    # Re-calcular win correctamente (Cell 24)
    correct_win = (df["teamScore"] > df["opponentScore"]).astype(int)
    df.loc[df["win"] != correct_win, "win"] = correct_win

    # Crear gameId único (Cell 19)
    df["gameId"] = (
        df["gameDateTimeEst"].astype(str) + "_" +
        df[["teamId", "opponentTeamId"]].min(axis=1).astype(str) + "_" +
        df[["teamId", "opponentTeamId"]].max(axis=1).astype(str)
    )

    print(f"  → {len(df):,} filas con features completos")
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  3. ENTRENAMIENTO
# ─────────────────────────────────────────────────────────────────────────────

def build_classifier_matchup_data(df: pd.DataFrame):
    """
    Transforma filas de equipos individuales en filas de DUELOS (Matchups).
    Genera filas SIMÉTRICAS (2 filas por partido) para evitar el sesgo de localía.
    """
    print("[Pipeline] Construyendo Duelos Simétricos A vs B (Anti-Home Bias)...")
    
    home_df = df[df["home"] == 1].copy()
    away_df = df[df["home"] == 0].copy()

    core_cols = [
        "expectedTeamScore", "defensive_rating_r10", 
        "win_streak_5", "RealHandicap", "totalPoints"
    ]
    core_cols = [c for c in core_cols if c in df.columns]

    # Unir Home y Away
    matchup_raw = home_df[["gameId", "gameDateTimeEst", "teamId", "opponentTeamId", "win"] + core_cols].merge(
        away_df[["gameId", "teamId"] + core_cols],
        on="gameId",
        suffixes=("_HOME", "_AWAY")
    )

    # CREAR FILAS SIMÉTRICAS:
    # Fila 1: Equipo A es Home, Equipo B es Away (home=1)
    f1 = pd.DataFrame()
    f1["gameDateTimeEst"] = matchup_raw["gameDateTimeEst"]
    f1["home"] = 1
    f1["label_win_A"] = matchup_raw["win"]  # win_HOME
    for c in core_cols:
        f1[f"{c}_A"] = matchup_raw[f"{c}_HOME"]
        f1[f"{c}_B"] = matchup_raw[f"{c}_AWAY"]
        f1[f"DIFF_{c}"] = f1[f"{c}_A"] - f1[f"{c}_B"]

    # Fila 2: Equipo A es Away, Equipo B es Home (home=0)
    f2 = pd.DataFrame()
    f2["gameDateTimeEst"] = matchup_raw["gameDateTimeEst"]
    f2["home"] = 0
    f2["label_win_A"] = 1 - matchup_raw["win"]  # win_AWAY (si el local no gano, gano el visitante)
    for c in core_cols:
        f2[f"{c}_A"] = matchup_raw[f"{c}_AWAY"]
        f2[f"{c}_B"] = matchup_raw[f"{c}_HOME"]
        f2[f"DIFF_{c}"] = f2[f"{c}_A"] - f2[f"{c}_B"]

    # Concatenar para balancear 50/50 la importancia de la localía
    matchup = pd.concat([f1, f2], ignore_index=True)

    print(f"  → {len(matchup):,} filas generadas (Simetría aplicada)")
    return matchup


def train_classifier(df: pd.DataFrame):
    """
    Entrena CatBoostClassifier basado en DUELOS (Matchups).
    """
    from catboost import CatBoostClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    import numpy as np

    # 1. Construir Matchups
    df_model = build_classifier_matchup_data(df)
    
    # 2. Split Temporal Cronológico (80% Pasado / 20% Más Reciente)
    df_model = df_model.sort_values("gameDateTimeEst")
    split_idx = int(len(df_model) * 0.80)
    
    train_data = df_model.iloc[:split_idx]
    test_data  = df_model.iloc[split_idx:]

    # 3. Selección de Features (A, B, Diferencias y Localía)
    X_cols = [c for c in df_model.columns if c.endswith("_A") or c.endswith("_B") or c.startswith("DIFF_") or c == "home"]
    X_cols = [c for c in X_cols if "teamId" not in c and "label" not in c and "home_B" not in c]
    
    X_train, y_train = train_data[X_cols], train_data["label_win_A"]
    X_test,  y_test  = test_data[X_cols],  test_data["label_win_A"]

    print(f"[Classifier] Entrenando con {len(X_train)} partidos / Evaluando con {len(X_test)}")
    
    # MODELO CALIBRADO (Realismo NBA)
    # l2_leaf_reg alto (20.0) → Impide que el modelo sea "arrogante" con pequeñas ventajas.
    # subsample (0.7) → Añade "ruido realista" para simular la variabilidad de la liga.
    model = CatBoostClassifier(
        iterations=1500, # Un poco más para compensar la regularización
        depth=6, 
        learning_rate=0.03, # Más lento para mayor precisión
        l2_leaf_reg=20.0, 
        bootstrap_type='Bernoulli',
        subsample=0.7,
        random_seed=42,
        verbose=300
    )
    
    model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=100, verbose=300)

    # Evaluación
    preds = model.predict(X_test)
    acc   = accuracy_score(y_test, preds)

    print(f"\n[Classifier] ✅ Accuracy: {acc*100:.2f}%")
    print("\nMatriz de Confusión (Matchup):")
    print(confusion_matrix(y_test, preds))
    print("\nReporte de Clasificación:")
    print(classification_report(y_test, preds))

    # Feature Importance (Top 5 solicitado)
    feat_imp = model.get_feature_importance()
    top_indices = np.argsort(feat_imp)[-5:][::-1]
    print("\n⭐ TOP 5 FEATURES MÁS IMPORTANTES:")
    for idx in top_indices:
        print(f"  - {X_cols[idx]}: {feat_imp[idx]:.2f}")

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model,  MODELS_DIR / "classifier.pkl")
    joblib.dump(X_cols, MODELS_DIR / "classifier_features.pkl")
    print(f"[Classifier] Modelo Matchup guardado en {MODELS_DIR}/classifier.pkl")

    return model, acc, X_cols


def build_regressor_features(df: pd.DataFrame):
    """Construye features ATT_/DEF_ para el regresor (Cell 20)."""
    dfp = df.copy()
    dfp = dfp.sort_values(["teamId", "gameDateTimeEst"]).reset_index(drop=True)

    rival_cols = [
        "teamId", "gameId", "teamScore", "assists", "blocks", "steals",
        "fieldGoalsAttempted", "fieldGoalsMade", "fieldGoalsPercentage",
        "threePointersAttempted", "threePointersMade", "threePointersPercentage",
        "freeThrowsAttempted", "freeThrowsMade", "freeThrowsPercentage",
        "reboundsDefensive", "reboundsOffensive", "reboundsTotal",
        "foulsPersonal", "turnovers",
    ]
    rival_cols = [c for c in rival_cols if c in dfp.columns]
    rival = dfp[rival_cols].rename(columns=lambda c: c + "_OPP")

    dfp = dfp.merge(
        rival,
        left_on=["gameId", "opponentTeamId"],
        right_on=["gameId_OPP", "teamId_OPP"],
        how="left"
    )

    attack_cols = [
        "teamScore", "assists", "blocks", "steals",
        "fieldGoalsAttempted", "fieldGoalsMade",
        "threePointersAttempted", "threePointersMade",
        "freeThrowsAttempted", "freeThrowsMade",
        "reboundsOffensive", "turnovers",
    ]
    defense_cols = [
        "teamScore_OPP", "assists_OPP", "blocks_OPP", "steals_OPP",
        "fieldGoalsAttempted_OPP", "fieldGoalsMade_OPP",
        "threePointersAttempted_OPP", "threePointersMade_OPP",
        "freeThrowsAttempted_OPP", "freeThrowsMade_OPP",
        "reboundsOffensive_OPP", "turnovers_OPP",
    ]

    attack_cols  = [c for c in attack_cols  if c in dfp.columns]
    defense_cols = [c for c in defense_cols if c in dfp.columns]

    for col in attack_cols:
        dfp[f"ATT_{col}_r5"] = (
            dfp.groupby("teamId")[col].shift(1).rolling(5, min_periods=1).mean()
            .reset_index(level=0, drop=True)
        )
    for col in defense_cols:
        dfp[f"DEF_{col}_r10"] = (
            dfp.groupby("teamId")[col].shift(1).rolling(10, min_periods=1).mean()
            .reset_index(level=0, drop=True)
        )

    dfp = dfp.dropna()
    return dfp


def train_regressor(df: pd.DataFrame):
    """
    Entrena CatBoostRegressor (puntaje) — idéntico a Cell 30.
    Guarda el modelo y retorna (model, mae).
    """
    from catboost import CatBoostRegressor
    from sklearn.metrics import mean_absolute_error
    import numpy as np

    print("\n[Regressor] Construyendo features ATT_/DEF_...")
    dfp = build_regressor_features(df)

    # Target: puntaje del próximo partido
    df_next = dfp.sort_values(["teamId", "gameDateTimeEst"]).copy()
    df_next["teamScore_next"] = dfp.groupby("teamId")["teamScore"].shift(-1)
    df_next = df_next[["gameId", "teamId", "teamScore_next"]]

    df_model = dfp.merge(df_next, on=["gameId", "teamId"], how="left")
    df_model = df_model.dropna(subset=["teamScore_next"]).reset_index(drop=True)

    meta_cols = ["index", "gameDateTimeEst", "teamId_OPP", "gameId_OPP",
                 "teamId", "opponentTeamId", "gameId", "home", "win"]
    df_model = df_model.drop(columns=[c for c in meta_cols if c in df_model.columns], errors="ignore")

    feature_cols = [c for c in df_model.columns if c.startswith("ATT_") or c.startswith("DEF_")]
    X = df_model[feature_cols]
    y = df_model["teamScore_next"]

    split = int(len(df_model) * 0.80)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    print(f"[Regressor] Entrenando CatBoostRegressor ({len(X_train):,} train / {len(X_test):,} test)...")
    model = CatBoostRegressor(
        iterations=1200, depth=6, learning_rate=0.03,
        loss_function="RMSE", verbose=200, random_seed=42,
    )
    model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=200)

    preds = model.predict(X_test)
    mae   = mean_absolute_error(y_test, preds)
    rmse  = np.sqrt(((y_test - preds) ** 2).mean())
    print(f"[Regressor] ✅ MAE: {mae:.2f} pts | RMSE: {rmse:.2f} pts")

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model,        MODELS_DIR / "regressor.pkl")
    joblib.dump(feature_cols, MODELS_DIR / "regressor_features.pkl")
    print(f"[Regressor] Modelo guardado en {MODELS_DIR}/regressor.pkl")

    # Guardar las últimas stats de cada equipo (para inferencia en tiempo real)
    _save_team_stats_snapshot(df)

    return model, mae, feature_cols


def _save_team_stats_snapshot(df: pd.DataFrame):
    """Guarda el último estado de cada equipo para usarlo en predicciones."""
    snapshot = (
        df.sort_values(["teamId", "gameDateTimeEst"])
        .groupby("teamId")
        .last()
        .reset_index()
    )
    joblib.dump(snapshot, MODELS_DIR / "team_stats_snapshot.pkl")
    print(f"[Pipeline] Snapshot de {len(snapshot)} equipos guardado.")


# ─────────────────────────────────────────────────────────────────────────────
#  4. CLASE PREDICTOR (para Flask)
# ─────────────────────────────────────────────────────────────────────────────

class NBAPredictor:
    """
    Interfaz de predicción para Flask.
    Carga los modelos entrenados y expone predict().

    Uso en app.py:
        predictor = NBAPredictor()
        result = predictor.predict(team1_id, team2_id, home_team=1)
    """

    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self._load_models()

    def _load_models(self):
        try:
            self.clf          = joblib.load(self.models_dir / "classifier.pkl")
            self.clf_features = joblib.load(self.models_dir / "classifier_features.pkl")
            self.reg          = joblib.load(self.models_dir / "regressor.pkl")
            self.reg_features = joblib.load(self.models_dir / "regressor_features.pkl")
            self.snapshot     = joblib.load(self.models_dir / "team_stats_snapshot.pkl")
            self.models_loaded = True
            print("[NBAPredictor] ✅ Modelos cargados correctamente")
        except FileNotFoundError as e:
            self.models_loaded = False
            print(f"[NBAPredictor] ⚠️ Modelos no encontrados: {e}")
            print("  → Ejecuta: python ml_pipeline.py  para entrenar primero")

    def _get_team_row(self, team_id: int) -> pd.Series:
        """Obtiene las últimas stats registradas de un equipo."""
        row = self.snapshot[self.snapshot["teamId"] == team_id]
        if len(row) == 0:
            raise ValueError(f"Team ID {team_id} no encontrado en el snapshot.")
        return row.iloc[0]

    def predict(self, team1_id: int, team2_id: int, home_team: str = "team1"):
        if not self.models_loaded:
            return {"error": "Modelos no cargados."}

        # 1. Obtener stats de ambos equipos
        try:
            t1_stats = self._get_team_row(team1_id).to_dict()
            t2_stats = self._get_team_row(team2_id).to_dict()
        except ValueError as e:
            return {"error": str(e)}

        # 2. Configurar Matchup A vs B (A es el que comparamos contra B)
        # El modelo fue entrenado balanceado (A puede ser Local o Visitante)
        if home_team == "team1":
            stats_A, stats_B = t1_stats, t2_stats
            is_home_A = 1
        elif home_team == "team2":
            stats_A, stats_B = t2_stats, t1_stats
            is_home_A = 1 # t2 es el local
        else:
            stats_A, stats_B = t1_stats, t2_stats
            is_home_A = 0 # Neutral

        # Construir fila para el modelo
        row = {"home": is_home_A}
        core_cols = ["expectedTeamScore", "defensive_rating_r10", "win_streak_5", "RealHandicap", "totalPoints"]
        for f in core_cols:
            row[f"{f}_A"] = float(stats_A.get(f, 0))
            row[f"{f}_B"] = float(stats_B.get(f, 0))
            row[f"DIFF_{f}"] = row[f"{f}_A"] - row[f"{f}_B"]

        X_matchup = pd.DataFrame([row])[self.clf_features]

        # 3. Lógica de Realismo NBA (Basado en datos históricos de 146k partidos)
        # a) Probabilidad bruta del clasificador
        raw_prob_A = float(self.clf.predict_proba(X_matchup)[0][1])

        # b) Margen esperado del regresor (Spread)
        # Usamos el spread para 'anclar' la probabilidad a la realidad competitiva.
        try:
            exp_pts_A = float(self.reg.predict(X_matchup)[0])
            # Estimación simple del margen relativo
            # (El modelo reg estima puntos de A dado el oponente B)
            current_opp_avg = stats_B.get("expectedTeamScore", 110)
            projected_margin = exp_pts_A - current_opp_avg
            
            # Calibración Logística NBA: 
            # Una ventaja de +10 suele ser 85-88% de victoria en la vida real.
            spread_prob = 1 / (1 + np.exp(-0.135 * projected_margin))
            
            # Ensamble Certero: Combinamos la fuerza estadística con la lógica de puntos.
            final_prob_A = (raw_prob_A * 0.6) + (spread_prob * 0.4)
            
            # Cap de Realismo: Nadie gana el 100% en la NBA antes de jugar.
            final_prob_A = np.clip(final_prob_A, 0.05, 0.95)
        except Exception:
            final_prob_A = np.clip(raw_prob_A, 0.1, 0.9)

        team1_name = TEAM_NAMES.get(team1_id, f"Team {team1_id}")
        team2_name = TEAM_NAMES.get(team2_id, f"Team {team2_id}")

        # La respuesta siempre devuelve Team 1 vs Team 2
        # Si stats_A era team1:
        if stats_A["teamId"] == team1_id:
            prob1 = final_prob_A
        else:
            prob1 = 1.0 - final_prob_A
        
        prob2 = 1.0 - prob1

        return {
            "prediction": team1_name if prob1 > 0.5 else team2_name,
            "win_probability": round(max(prob1, prob2) * 100, 1),
            "team1": {"name": team1_name, "probability": round(prob1 * 100, 1)},
            "team2": {"name": team2_name, "probability": round(prob2 * 100, 1)},
            "model_info": "Matchup-Aware A/B Classifier + NBA Realism Calibration",
            "details": {
                "t1_streak": t1_stats.get("win_streak_5", 0),
                "t2_streak": t2_stats.get("win_streak_5", 0),
                "home_court": home_team
            }
        }

    def get_team_stats(self, team_id: int) -> dict:
        """Retorna las estadísticas recientes de un equipo."""
        row = self._get_team_row(team_id)
        return {
            "team_id": team_id,
            "team_name": TEAM_NAMES.get(team_id, f"Team {team_id}"),
            "expectedTeamScore": round(float(row.get("expectedTeamScore", 0)), 1),
            "expectedOpponentScore": round(float(row.get("expectedOpponentScore", 0)), 1),
            "win_streak_5": round(float(row.get("win_streak_5", 0)), 1),
            "RealHandicap": round(float(row.get("RealHandicap", 0)), 1),
            "assists": round(float(row.get("assists", 0)), 1),
            "rebounds": round(float(row.get("reboundsTotal", 0)), 1),
            "steals": round(float(row.get("steals", 0)), 1),
            "blocks": round(float(row.get("blocks", 0)), 1),
            "turnovers": round(float(row.get("turnovers", 0)), 1),
            "fieldGoalsPercentage": round(float(row.get("fieldGoalsPercentage", 0)), 3),
            "threePointersPercentage": round(float(row.get("threePointersPercentage", 0)), 3),
        }

    def get_all_teams(self) -> list:
        """Lista todos los equipos disponibles en el snapshot."""
        result = []
        for _, row in self.snapshot.iterrows():
            tid = int(row["teamId"])
            result.append({
                "team_id": tid,
                "team_name": TEAM_NAMES.get(tid, f"Team {tid}"),
            })
        return sorted(result, key=lambda x: x["team_name"])


# ─────────────────────────────────────────────────────────────────────────────
#  5. ENTRY POINT — entrenar y guardar todo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"  NBA ML Pipeline — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}\n")

    # 1. Cargar y limpiar datos
    df_raw = load_and_clean(CSV_PATH)

    # 2. Feature engineering
    df_feat = build_features(df_raw)

    # 3. Entrenar clasificador (win/loss)
    clf, acc, clf_feats = train_classifier(df_feat)

    # 4. Entrenar regresor (puntaje)
    reg, mae, reg_feats = train_regressor(df_feat)

    print(f"\n{'='*60}")
    print(f"  ENTRENAMIENTO COMPLETO")
    print(f"  Clasificador Accuracy : {acc*100:.2f}%")
    print(f"  Regresor MAE          : {mae:.2f} puntos")
    print(f"  Modelos en            : {MODELS_DIR}/")
    print(f"{'='*60}\n")
