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
CSV_PATH   = "games.csv"
MODELS_DIR = Path("models")

# IDs a eliminar (del notebook original)
IDS_A_ELIMINAR = [
    15016, 15018, 50013, 50014,
    1610612737, 1610612764, 1610612758,
    1610612755, 1610612746, 1610612766, 1610612760
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

    # Fechas (Cell 4)
    df["gameDateTimeEst"] = pd.to_datetime(
        df["gameDateTimeEst"], errors="coerce", format="mixed", utc=True
    ).dt.normalize()
    df = df[(df["gameDateTimeEst"] >= "2015-01-01") & (df["gameDateTimeEst"] <= "2025-12-31")]
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
    df["win"] = df["win"].astype(int)

    # Ordenar por equipo y fecha
    df = df.sort_values(["teamId", "gameDateTimeEst"]).reset_index(drop=True)

    # Win streak (Cell 14)
    df["win_streak_5"] = (
        (1 - df["win"])
        .groupby(df["teamId"])
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

def train_classifier(df: pd.DataFrame):
    """
    Entrena CatBoostClassifier (win/loss) — idéntico a Cell 29.
    Guarda el modelo y retorna (model, accuracy).
    """
    from catboost import CatBoostClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    print("\n[Classifier] Entrenando CatBoostClassifier...")

    features = [f for f in CLASSIFIER_FEATURES if f in df.columns]
    X = df[features]
    y = df["win"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = CatBoostClassifier(
        iterations=1200,
        learning_rate=0.045,
        depth=6,
        eval_metric="Accuracy",
        random_seed=42,
        verbose=200,
        use_best_model=True,
    )
    model.fit(X_train, y_train, eval_set=(X_test, y_test))

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[Classifier] ✅ Accuracy: {acc*100:.2f}%")

    # Guardar modelo y metadatos
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model,    MODELS_DIR / "classifier.pkl")
    joblib.dump(features, MODELS_DIR / "classifier_features.pkl")
    print(f"[Classifier] Modelo guardado en {MODELS_DIR}/classifier.pkl")

    return model, acc, features


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

    def predict(self, team1_id: int, team2_id: int, home_team: str = "team1") -> dict:
        """
        Predice el resultado de un partido.

        Args:
            team1_id:  NBA team ID del equipo 1
            team2_id:  NBA team ID del equipo 2
            home_team: "team1" | "team2" | "neutral"

        Returns:
            dict con probabilidades, puntaje proyectado y factores del modelo.
        """
        if not self.models_loaded:
            raise RuntimeError("Modelos no cargados. Ejecuta ml_pipeline.py primero.")

        t1 = self._get_team_row(team1_id)
        t2 = self._get_team_row(team2_id)

        # Construir vector de features del equipo 1 (usa sus propias stats históricas)
        home_bonus = 2.5 if home_team == "team1" else (-2.5 if home_team == "team2" else 0)

        def make_features(team_row, opp_row, is_home: bool) -> pd.DataFrame:
            row = {}
            for feat in self.clf_features:
                if feat in team_row.index:
                    row[feat] = team_row[feat]
                else:
                    row[feat] = 0.0
            # Ajuste por ventaja de local
            if is_home and "expectedTeamScore" in row:
                row["expectedTeamScore"] = row.get("expectedTeamScore", 0) + 2.5
            return pd.DataFrame([row])

        X1 = make_features(t1, t2, home_team == "team1")
        X2 = make_features(t2, t1, home_team == "team2")

        # Clasificador: probabilidad de victoria
        prob1 = float(self.clf.predict_proba(X1)[0][1])
        prob2 = 1 - prob1

        # Regresor: puntaje proyectado
        # Usamos features ATT_/DEF_ si están disponibles, sino estimamos
        score1_raw = float(t1.get("expectedTeamScore", 110))
        score2_raw = float(t2.get("expectedTeamScore", 110))

        if home_team == "team1":
            score1_raw += 2.5
        elif home_team == "team2":
            score2_raw += 2.5

        # Feature importances del clasificador
        importances = {}
        feat_imp = self.clf.get_feature_importance()
        for feat, imp in zip(self.clf_features, feat_imp):
            importances[feat] = round(float(imp), 2)

        top_features = sorted(importances.items(), key=lambda x: -x[1])[:5]

        return {
            "team1_id": team1_id,
            "team2_id": team2_id,
            "team1_name": TEAM_NAMES.get(team1_id, f"Team {team1_id}"),
            "team2_name": TEAM_NAMES.get(team2_id, f"Team {team2_id}"),
            "team1_win_prob": round(prob1 * 100, 1),
            "team2_win_prob": round(prob2 * 100, 1),
            "team1_projected_score": round(score1_raw, 1),
            "team2_projected_score": round(score2_raw, 1),
            "home_team": home_team,
            "model": "CatBoostClassifier",
            "top_features": [{"feature": f, "importance": i} for f, i in top_features],
            "team1_recent": {
                "win_streak_5": round(float(t1.get("win_streak_5", 0)), 1),
                "expectedTeamScore": round(float(t1.get("expectedTeamScore", 0)), 1),
                "RealHandicap": round(float(t1.get("RealHandicap", 0)), 1),
            },
            "team2_recent": {
                "win_streak_5": round(float(t2.get("win_streak_5", 0)), 1),
                "expectedTeamScore": round(float(t2.get("expectedTeamScore", 0)), 1),
                "RealHandicap": round(float(t2.get("RealHandicap", 0)), 1),
            },
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
