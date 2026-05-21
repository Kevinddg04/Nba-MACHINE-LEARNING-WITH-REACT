"""
ml_pipeline.py
==============
Flujo completo de Machine Learning para predicciones de la NBA.
Adaptado para la automatización diaria en la nube (CatBoost + LightGBM).

Contiene:
  - Preprocesamiento de datos y Feature Engineering defensivo/ofensivo.
  - Clasificador (CatBoost + LightGBM)  → Predice GANADOR / PERDEDOR
  - Regresor (CatBoost)                 → Estima el ANOTAJE ESPERADO
  - Módulos de Calibración de probabilidad de victoria usando la lógica de los puntos.

USO:
    python ml_pipeline.py
"""

import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURACIÓN GLOBAL
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent

CSV_PATH   = str(PROJECT_ROOT / "TeamStatistics.csv")
MODELS_DIR = BASE_DIR / "models"

# IDs ignorados (Partidos de exhibición o All-Star que distorsionan el modelo)
IDS_A_ELIMINAR = [
    15016, 15018, 50013, 50014
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
#  1. CARGA Y LIMPIEZA
# ─────────────────────────────────────────────────────────────────────────────

def load_and_clean(csv_path: str = CSV_PATH) -> pd.DataFrame:
    """Carga el dataset CSV y aplica limpiadores iniciales anti-ruido."""
    print("[Pipeline] Cargando base de datos histórica de partidos NBA...")
    df = pd.read_csv(csv_path, index_col=0, low_memory=False)
    print(f"  → {len(df):,} registros cargados exitosamente.")

    # Columnas irrelevantes para la IA predictiva
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

    # Restricción Temporal de Entrenamiento (Ignorar eras pasadas irrelevantes para el basket moderno)
    df["gameDateTimeEst"] = pd.to_datetime(
        df["gameDateTimeEst"], errors="coerce", format="mixed", utc=True
    ).dt.normalize()
    # Entrenar desde el 2015 en adelante (La Era de los Triples) hasta 2026/Presente
    df = df[(df["gameDateTimeEst"] >= "2015-01-01") & (df["gameDateTimeEst"] <= "2026-12-31")]
    df = df.sort_values("gameDateTimeEst").reset_index(drop=True)

    # Limpiar identificadores ficticios o All-Star
    valid_ids = set(df["teamId"].unique()) | set(df["opponentTeamId"].unique())
    df = df[
        df["teamId"].isin(valid_ids) &
        df["opponentTeamId"].isin(valid_ids)
    ].copy()
    df = df[
        ~(df["teamId"].isin(IDS_A_ELIMINAR) | df["opponentTeamId"].isin(IDS_A_ELIMINAR))
    ].reset_index(drop=True)

    print(f"  → {len(df):,} registros aceptados tras la exclusión de ruido.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  2. CREACIÓN DE VARIABLES AVANZADAS (FEATURE ENGINEERING)
# ─────────────────────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extrae las rachas (streaks), handicaps reales y promedios continuos (rolling)."""
    print("[Pipeline] Iniciando Ingeniería de Datos Deportivos (Feature Engineering)...")

    # Forzar etiqueta binaria de Victoria (Win)
    df = df.dropna(subset=["win"]).copy()
    df["win"] = df["win"].astype(int)

    # Ordenar cronológicamente todo el torneo antes de promediar
    df = df.sort_values(["teamId", "gameDateTimeEst"]).reset_index(drop=True)

    # Racha de victorias en los últimos 5 partidos (Win Streak)
    df["win_streak_5"] = (
        df.groupby("teamId")["win"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum())
    )

    # Total de Puntos Reales del Duelo
    df["totalPoints_real"] = df["teamScore"] + df["opponentScore"]

    # Prevención de Espionaje Temporal (Leakage): Desplazar métricas 1 partido atrás
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

    # Promediadores (Rolling Means) de los 5 juegos previos
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

    # Defectos y Handicaps Promedios
    df["PointDiff"] = df["teamScore"] - df["opponentScore"]
    df["Handicap"]  = df["PointDiff"] * -1

    # Puntajes Ofensivos Promedio (Expected Scores)
    df["expectedTeamScore"] = df.groupby("teamId")["teamScore"].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )
    df["expectedOpponentScore"] = df.groupby("teamId")["opponentScore"].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )
    df["totalPoints"] = df["expectedTeamScore"] + df["expectedOpponentScore"]

    # Ventaja Real en la Cancha (Handicap Promedio)
    df["RealHandicap"] = df["teamScore"] - df["opponentScore"]
    df["RealHandicap"] = (
        df.groupby("teamId")["RealHandicap"]
        .apply(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )

    # Evaluación Defensiva: ¿Cuántos puntos permite este equipo en los últimos 10 juegos?
    df["defensive_rating_r10"] = (
        df.groupby("teamId")["opponentScore"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    )

    df = df.dropna().reset_index(drop=True)

    # Re-calcular victoria en caso de inconsistencia en CSV original
    correct_win = (df["teamScore"] > df["opponentScore"]).astype(int)
    df.loc[df["win"] != correct_win, "win"] = correct_win

    # Crear Identificador Universal de Partido (Evita que el equipo local/visitante tengan IDs diferentes)
    df["gameId"] = (
        df["gameDateTimeEst"].astype(str) + "_" +
        df[["teamId", "opponentTeamId"]].min(axis=1).astype(str) + "_" +
        df[["teamId", "opponentTeamId"]].max(axis=1).astype(str)
    )

    print(f"  → {len(df):,} registros enriquecidos listos para la IA.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  3. ENTRENAMIENTO DEL MODELO DE APRENDIZAJE AUTOMÁTICO (MACHINE LEARNING)
# ─────────────────────────────────────────────────────────────────────────────

def build_classifier_matchup_data(df: pd.DataFrame):
    """
    Transforma registros estadísticos simples en una batalla 1vs1.
    Genera filas SIMÉTRICAS A vs B y B vs A para suprimir sesgos de Localía ilusoria.
    """
    print("[Clasificador] Construyendo Arquitectura Simétrica de Duelos (Cero Sesgos Localistas)...")
    
    home_df = df[df["home"] == 1].copy()
    away_df = df[df["home"] == 0].copy()

    core_cols = [
        "expectedTeamScore", "defensive_rating_r10", 
        "win_streak_5", "RealHandicap", "totalPoints"
    ]
    core_cols = [c for c in core_cols if c in df.columns]

    matchup_raw = home_df[["gameId", "gameDateTimeEst", "teamId", "opponentTeamId", "win"] + core_cols].merge(
        away_df[["gameId", "teamId"] + core_cols],
        on="gameId",
        suffixes=("_HOME", "_AWAY")
    )

    # Perspectiva 1: A es Local, B es Visitante
    f1 = pd.DataFrame()
    f1["gameDateTimeEst"] = matchup_raw["gameDateTimeEst"]
    f1["home"] = 1
    f1["label_win_A"] = matchup_raw["win"]  # La victoria del local
    for c in core_cols:
        f1[f"{c}_A"] = matchup_raw[f"{c}_HOME"]
        f1[f"{c}_B"] = matchup_raw[f"{c}_AWAY"]
        f1[f"DIFF_{c}"] = f1[f"{c}_A"] - f1[f"{c}_B"]

    # Perspectiva 2: A es Visitante, B es Local (El espejo)
    f2 = pd.DataFrame()
    f2["gameDateTimeEst"] = matchup_raw["gameDateTimeEst"]
    f2["home"] = 0
    f2["label_win_A"] = 1 - matchup_raw["win"]  # Reflejo inverso de victoria
    for c in core_cols:
        f2[f"{c}_A"] = matchup_raw[f"{c}_AWAY"]
        f2[f"{c}_B"] = matchup_raw[f"{c}_HOME"]
        f2[f"DIFF_{c}"] = f2[f"{c}_A"] - f2[f"{c}_B"]

    # Mezclar ambas perspectivas
    matchup = pd.concat([f1, f2], ignore_index=True)
    print(f"  → {len(matchup):,} escenarios de combate generados y balanceados.")
    return matchup


def train_classifier(df: pd.DataFrame):
    """
    Entrena el motor lógico principal para predecir Qué Equipo Ganará.
    Utiliza una aleación de CatBoost y LightGBM (Ensemble).
    """
    from catboost import CatBoostClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    import numpy as np

    df_model = build_classifier_matchup_data(df)
    
    # Simular Reto en el Mundo Real: Entrenar con el 80% viejo, adivinar el 20% futuro
    df_model = df_model.sort_values("gameDateTimeEst")
    split_idx = int(len(df_model) * 0.80)
    
    train_data = df_model.iloc[:split_idx]
    test_data  = df_model.iloc[split_idx:]

    # Aislar unicamente las métricas matemáticas
    X_cols = [c for c in df_model.columns if c.endswith("_A") or c.endswith("_B") or c.startswith("DIFF_") or c == "home"]
    X_cols = [c for c in X_cols if "teamId" not in c and "label" not in c and "home_B" not in c]
    
    X_train_full, y_train = train_data[X_cols], train_data["label_win_A"]
    X_test_full,  y_test  = test_data[X_cols],  test_data["label_win_A"]

    from ml.ensemble import build_ensemble
    from ml.feature_selection import select_features_final
    from ml.calibration import calibrate_classifier
    from ml.cross_validation import time_series_cv

    print(f"[Clasificador] Exigiendo destilación de las {len(X_cols)} métricas principales...")
    selected_cols = select_features_final(X_train_full, y_train, k=35, vif_threshold=15.0)
    
    # La Cancha Local es inquebrantable en NBA
    if "home" not in selected_cols and "home" in X_train_full.columns:
        selected_cols.append("home")
        
    X_train = X_train_full[selected_cols]
    X_test  = X_test_full[selected_cols]

    print(f"[Clasificador] Estudiando {len(X_train):,} eventos pasados para dominar el futuro.")
    
    ensemble = build_ensemble(
        catboost_params={'iterations': 600, 'depth': 6, 'learning_rate': 0.03, 'l2_leaf_reg': 20.0, 'bootstrap_type': 'Bernoulli', 'subsample': 0.7, 'random_seed': 42, 'verbose': False},
        lgbm_params={'n_estimators': 500, 'learning_rate': 0.03, 'max_depth': 6, 'reg_lambda': 10.0, 'subsample': 0.7, 'random_state': 42, 'verbose': -1}
    )

    print("[Clasificador] Calibrando porcentajes de victoria (Asignador de Probabilidad) ...")
    calibrated_model = calibrate_classifier(ensemble, X_train, y_train, cv=3)

    preds = calibrated_model.predict(X_test)
    acc   = accuracy_score(y_test, preds)

    print(f"\n[Clasificador] ✅ Precisión Oficial del Motor Principal: {acc*100:.2f}%")
    print("\n[Métricas] Desglose Interno del Cerebro:")
    print(classification_report(y_test, preds))

    try:
        cb = ensemble.named_estimators_['catboost']
        feat_imp = cb.get_feature_importance()
    except Exception:
        feat_imp = np.ones(len(selected_cols))
        
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(calibrated_model,  MODELS_DIR / "classifier.pkl")
    joblib.dump(selected_cols, MODELS_DIR / "classifier_features.pkl")
    joblib.dump(feat_imp, MODELS_DIR / "classifier_feature_importances.pkl")
    print(f"[Clasificador] Inteligencia artificial guardada exitosamente.")

    return calibrated_model, acc, selected_cols


def build_regressor_features(df: pd.DataFrame):
    """Crea columnas de Ataque (ATT) y Defensa (DEF) puras para predecir cuántos puntos meterá alguien."""
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
    Entrena el segundo cerebro (El Matemático de Puntajes).
    Predice el Spread (Brecha de anotaciones hipotética).
    """
    from catboost import CatBoostRegressor
    from sklearn.metrics import mean_absolute_error
    import numpy as np

    print("\n[Regresor Matemático] Calculando brechas de capacidad de anotación...")
    dfp = build_regressor_features(df)

    # Objetivo a predecir (Target): ¿Cuántos puntos anotará en el futuro el equipo examinado?
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

    print(f"[Regresor Matemático] Memorizando puntajes de la liga ({len(X_train):,} eventos)...")
    model = CatBoostRegressor(
        iterations=1200, depth=6, learning_rate=0.03,
        loss_function="RMSE", verbose=200, random_seed=42,
    )
    # Por limitaciones de RAM y simplicidad de bitacora, minimizamos los mensajes de carga (verbose) a False
    model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=False)

    preds = model.predict(X_test)
    mae   = mean_absolute_error(y_test, preds)
    rmse  = np.sqrt(((y_test - preds) ** 2).mean())
    print(f"[Regresor Matemático] ✅ Error Absoluto de Estimación de Puntaje (MAE): {mae:.2f} puntos")

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model,        MODELS_DIR / "regressor.pkl")
    joblib.dump(feature_cols, MODELS_DIR / "regressor_features.pkl")
    print(f"[Regresor Matemático] Álgebra guardada exitosamente en el disco.")

    _save_team_stats_snapshot(df)

    return model, mae, feature_cols


def _save_team_stats_snapshot(df: pd.DataFrame):
    """Extrae las fotografías del "Último Minuto" para todos los equipos. La foto más fresca."""
    snapshot = (
        df.sort_values(["teamId", "gameDateTimeEst"])
        .groupby("teamId")
        .last()
        .reset_index()
    )
    joblib.dump(snapshot, MODELS_DIR / "team_stats_snapshot.pkl")
    print(f"[Pipeline] Memoria Reciente fotográfica capturada para {len(snapshot)} equipos. Guardado Exitoso.")


# ─────────────────────────────────────────────────────────────────────────────
#  4. SERVICIO PREDICTOR EN VIVO (Para Interfaz de React & API)
# ─────────────────────────────────────────────────────────────────────────────

class NBAPredictor:
    """Conserje de la Inteligencia Artificial. Responde las peticiones provenientes del Backend."""

    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self._load_models()

    def _load_models(self):
        """Intenta extraer los cerebros artificiales del disco a la memoria RAM."""
        try:
            self.clf          = joblib.load(self.models_dir / "classifier.pkl")
            self.clf_features = joblib.load(self.models_dir / "classifier_features.pkl")
            self.reg          = joblib.load(self.models_dir / "regressor.pkl")
            self.reg_features = joblib.load(self.models_dir / "regressor_features.pkl")
            self.snapshot     = joblib.load(self.models_dir / "team_stats_snapshot.pkl")
            try:
                self.clf_importances = joblib.load(self.models_dir / "classifier_feature_importances.pkl")
            except FileNotFoundError:
                self.clf_importances = np.ones(len(self.clf_features))
            self.models_loaded = True
            print("[Gestor Inteligente] ✅ Inteligencia recargada a la memoria con éxito. Estamos Listos.")
        except Exception as e:
            self.models_loaded = False
            print(f"[Gestor Inteligente] ⚠️ Los Modelos Artificiales aún no existen u ocurrió un error: {e}")
            print("  → Permite que el sistema complete automáticamente su primer aprendizaje diario.")

    def _get_team_row(self, team_id: int) -> pd.Series:
        """Obtiene la información fotográfica mental requerida para predecir a un equipo equis."""
        row = self.snapshot[self.snapshot["teamId"] == team_id]
        if len(row) == 0:
            raise ValueError(f"Equipo Numérico {team_id} no pudo ser encontrado en los registros.")
        return row.iloc[0]

    def predict(self, team1_id: int, team2_id: int, home_team: str = "team1"):
        """Deduce el ganador absoluto comparando las matemáticas de T1 contra las matemáticas de T2."""
        if not self.models_loaded:
            return {"error": "Cerebro Crítico no cargado aún. Permite que entrene primero."}

        try:
            t1_stats = self._get_team_row(team1_id).to_dict()
            t2_stats = self._get_team_row(team2_id).to_dict()
        except ValueError as e:
            return {"error": str(e)}

        if home_team == "team1":
            stats_A, stats_B = t1_stats, t2_stats
            is_home_A = 1
        elif home_team == "team2":
            stats_A, stats_B = t2_stats, t1_stats
            is_home_A = 1
        else:
            stats_A, stats_B = t1_stats, t2_stats
            is_home_A = 0 # Cancha Neutral

        row = {"home": is_home_A}
        core_cols = ["expectedTeamScore", "defensive_rating_r10", "win_streak_5", "RealHandicap", "totalPoints"]
        for f in core_cols:
            row[f"{f}_A"] = float(stats_A.get(f, 0))
            row[f"{f}_B"] = float(stats_B.get(f, 0))
            row[f"DIFF_{f}"] = row[f"{f}_A"] - row[f"{f}_B"]

        X_matchup = pd.DataFrame([row])[self.clf_features]

        # Aplicación estricta de la Realidad Deportiva (Realism Scaling)
        raw_prob_A = float(self.clf.predict_proba(X_matchup)[0][1])

        try:
            exp_pts_A = float(self.reg.predict(X_matchup)[0])
            current_opp_avg = stats_B.get("expectedTeamScore", 110)
            projected_margin = exp_pts_A - current_opp_avg
            
            # Margen Analítico: +10 puntos previstos = Fuerte Anclaje al 85% de Victoria Final.
            spread_prob = 1 / (1 + np.exp(-0.135 * projected_margin))
            
            # Equilibrio Armónico: 60% Análisis Técnico vs 40% Sensatez de Anotación
            final_prob_A = (raw_prob_A * 0.6) + (spread_prob * 0.4)
            # Acotado ético para proteger contra adivinanzas mágicas del 100% que faltan al respeto a la liga
            final_prob_A = np.clip(final_prob_A, 0.05, 0.95)
        except Exception:
            final_prob_A = np.clip(raw_prob_A, 0.1, 0.9)

        team1_name = TEAM_NAMES.get(team1_id, f"Equipo {team1_id}")
        team2_name = TEAM_NAMES.get(team2_id, f"Equipo {team2_id}")

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
            "model_info": "Clasificador Doble Avanzado + Calibración Práctica de Cancha 2.0",
            "details": {
                "t1_streak": t1_stats.get("win_streak_5", 0),
                "t2_streak": t2_stats.get("win_streak_5", 0),
                "home_court": home_team
            }
        }

    def get_team_stats(self, team_id: int) -> dict:
        """Sintetiza la ficha de un equipo demandada por los standings del Frontend."""
        row = self._get_team_row(team_id)
        return {
            "team_id": team_id,
            "team_name": TEAM_NAMES.get(team_id, f"Equipo {team_id}"),
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
        """Devuelve el registro oficial para encartar en la página web."""
        result = []
        for _, row in self.snapshot.iterrows():
            tid = int(row["teamId"])
            result.append({
                "team_id": tid,
                "team_name": TEAM_NAMES.get(tid, f"Equipo {tid}"),
            })
        return sorted(result, key=lambda x: x["team_name"])


# ─────────────────────────────────────────────────────────────────────────────
#  5. GATILLO DE ENTRENAMIENTO ESTRELLA
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n{'='*75}")
    print(f"  🧠 NBA Inteligencia Artificial (Retención Diaria) — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*75}\n")

    # 1. Absorber Historias
    df_raw = load_and_clean(CSV_PATH)

    # 2. Ingeniería Computacional de Atributos Deportivos
    df_feat = build_features(df_raw)

    # 3. Adiestrar Instinto de Victoria o Derrota
    clf, acc, clf_feats = train_classifier(df_feat)

    # 4. Adiestrar Perfección de la Aguja Anotadora
    reg, mae, reg_feats = train_regressor(df_feat)

    print(f"\n{'='*65}")
    print(f"  🏁 CUBÍCULO DE ENTRENAMIENTO CONCLUIDO CON ÉXITO")
    print(f"  Aciertos Oficiales Post-Calibración : {acc*100:.2f}% de Fidelidad")
    print(f"  Incertidumbre Absoluta General      : {mae:.2f} Puntos (MAE)")
    print(f"  Modelos Protegidos En Bóveda       : {MODELS_DIR}/")
    print(f"{'='*65}\n")
