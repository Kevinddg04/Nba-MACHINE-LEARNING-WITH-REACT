"""
main.py
=======
Backend API usando FastAPI. 
Reemplaza al antiguo app.py (Flask).
Incorpora Pydantic nativo, Redis caché y Background Tasks.
"""

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError
import os
import random
import subprocess
import requests as http_requests

from config.logging_config import get_logger
from ml_pipeline import NBAPredictor, TEAM_NAMES
from schemas import PredictionRequest, HeadToHeadRequest, AdminUpdateRequest
from database import SessionLocal, engine
from database.models import Base, PredictionLog
from metrics.calculator import calculate_hit_rate, get_last_predictions
from cache import cache_response, REDIS_AVAILABLE

logger = get_logger(__name__)

# Configuración Inicial de la DB
try:
    Base.metadata.create_all(bind=engine)
    logger.info("Base de datos inicializada correctamente")
except Exception as e:
    logger.error(f"Error inicializando DB de auditoría: {e}")

app = FastAPI(
    title="NBA ML Predictor v2.0",
    description="Predicciones de baloncesto usando CatBoost+LightGBM Ensemble, y caché en Redis",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = NBAPredictor()

# --- Configuración Constantes ---
PORT = int(os.environ.get("PORT", 8000))
SELF_URL = os.environ.get("SELF_URL", f"http://localhost:{PORT}")
ADMIN_SECRET = os.environ.get("ADMIN_SECRET", "super-secret-local-key")

# Dependencia DB
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def save_prediction_db(db, t1: int, t2: int, home: str, result: dict):
    """Guarda a DB de manera síncrona/segura."""
    try:
        import json
        log = PredictionLog(
            team1_id=t1,
            team2_id=t2,
            home_team=home,
            predicted_winner=result["prediction"],
            win_probability=result["win_probability"],
            model_version=result.get("model_version", "2.0"),
            metadata_json=json.dumps(result.get("details", {}))
        )
        db.add(log)
        db.commit()
    except Exception as e:
        logger.error(f"Fallo al auditar predicción en base de datos: {e}")
        db.rollback()


# --- Exception Handlers ---
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Convierte los errores de Pydantic al formato anterior esperado (pero con un dict o detail).
    Mantiene 422 que es estándar en FastAPI."""
    errors = exc.errors()
    msg = "; ".join([f"{err['loc'][-1] if err.get('loc') else 'Body'}: {err['msg']}" for err in errors])
    logger.warning(f"Validación fallida: {msg}")
    return JSONResponse(status_code=422, content={"error": msg})


@app.exception_handler(ValidationError)
async def pydantic_validation_exception_handler(request: Request, exc: ValidationError):
    """Por si usamos Pydantic puro internamente."""
    errors = exc.errors()
    msg = "; ".join([f"{err['loc'][-1] if err.get('loc') else 'Body'}: {err['msg']}" for err in errors])
    return JSONResponse(status_code=422, content={"error": msg})


# --- Tareas de Fondo en Segundo Plano ---
def run_update_pipeline():
    """Ejecuta el ciclo de entrenamiento de Machine Learning en segundo plano."""
    logger.info("Iniciando Entrenamiento de IA en segundo plano...")
    try:
        # 1. Traer datos
        subprocess.run(["python", "kaggle_fetcher.py"], check=True)
        # 2. Entrenar
        subprocess.run(["python", "ml_pipeline.py"], check=True)
        # 3. Cargar a la RAM
        predictor._load_models()
        logger.info("✅ Pipeline de Machine Learning Finalizado y recargado.")
    except Exception as e:
        logger.error(f"❌ Falló la actualización obligatoria del modelo: {e}")

def keep_alive_ping():
    """Auto-ping para mantener el servidor vivo."""
    try:
        logger.debug("Auto-ping de servidor ejecutándose...")
        http_requests.get(f"{SELF_URL}/health", timeout=5)
    except Exception as e:
        logger.warning(f"Fallo en el auto-ping de vida: {e}")

import asyncio
@app.on_event("startup")
async def start_keep_alive():
    """Evento disparado al momento de encender el servidor."""
    if os.environ.get("PORT"):
        asyncio.create_task(keep_alive_loop())
    
    # Arrancar el rastreador/entrenador eterno de Kaggle (cada 24h)
    asyncio.create_task(daily_learning_loop())

async def daily_learning_loop():
    """Bucle eterno que descarga datos frescos de Kaggle y re-entrena a la IA cada 24 horas."""
    while True:
        # Duerme por 24 horas exactas (86,400 segundos)
        await asyncio.sleep(86400)
        logger.info("Iniciando RUTINA DIARIA AUTOMÁTICA de Aprendizaje (Extracción Kaggle + Entrenamiento)...")
        try:
            # 1. Ejecutamos el agente de Kaggle en un subproceso para cuidar la memoria
            subprocess.run(["python", "kaggle_fetcher.py"], check=True)
            # 2. Ejecutamos el entrenamiento masivo
            subprocess.run(["python", "ml_pipeline.py"], check=True)
            # 3. Recargamos la inteligencia en el predictos actual
            predictor._load_models()
            logger.info("✅ Rutina Diaria de Aprendizaje Completada con Éxito. ¡El modelo es más inteligente ahora!")
        except Exception as e:
            logger.error(f"❌ Falló trágicamente el aprendizaje diario automático: {e}")

async def keep_alive_loop():
    import httpx
    async with httpx.AsyncClient() as client:
        while True:
            await asyncio.sleep(840)  # 14 minutos (evita suspensión de 15m en Render)
            try:
                await client.get(f"{SELF_URL}/health", timeout=10.0)
            except Exception as e:
                pass



# --- Endpoints API ---

@app.get("/health")
def health():
    return {
        "status": "ok", 
        "models_loaded": predictor.models_loaded, 
        "redis_available": REDIS_AVAILABLE
    }

@app.get("/ping")
def ping(background_tasks: BackgroundTasks):
    """Retorna pong y programa auto-ping futuro (si aplica)"""
    if os.environ.get("PORT"):
        background_tasks.add_task(keep_alive_ping)
    return {"status": "pong"}

def _get_teams_internal():
    """Internal helper to get teams without triggering cache wrapper as coroutine."""
    teams = predictor.get_all_teams()
    result = []
    
    WEST = {1610612742, 1610612743, 1610612744, 1610612745, 1610612746, 1610612747, 1610612763, 1610612750, 1610612740, 1610612760, 1610612756, 1610612757, 1610612758, 1610612759, 1610612762}
    
    for t in teams:
        tid = t["team_id"]
        row_df = predictor.snapshot[predictor.snapshot["teamId"] == tid]
        if row_df.empty:
            continue
        row = row_df.iloc[0]
        result.append({
            "team_id": tid,
            "name": t["team_name"],
            "conf": "West" if tid in WEST else "East",
            "expectedScore": round(float(row.get("expectedTeamScore", 0)), 1),
            "expectedOpponentScore": round(float(row.get("expectedOpponentScore", 0)), 1),
            "net_expected": round(float(row.get("expectedTeamScore", 0)) - float(row.get("expectedOpponentScore", 0)), 1),
            "win_streak_5": round(float(row.get("win_streak_5", 0)), 1),
            "RealHandicap": round(float(row.get("RealHandicap", 0)), 1),
            "fg_pct": round(float(row.get("fieldGoalsPercentage", 0)) * 100, 1),
        })
    return result

@app.get("/api/teams")
@cache_response(expire_seconds=3600)  # Cacha 1 hora
async def get_teams():
    if not predictor.models_loaded:
        raise HTTPException(status_code=503, detail="Modelos no cargados")
    result = _get_teams_internal()
    logger.info(f"GET /api/teams -> {len(result)}")
    return result

@app.get("/api/standings")
@cache_response(expire_seconds=3600)
async def get_standings(conf: str = None):
    if not predictor.models_loaded:
        raise HTTPException(status_code=503, detail="Modelos no cargados")
    
    teams = _get_teams_internal()
    if conf:
        teams = [t for t in teams if t["conf"].lower() == conf.lower()]
    
    # Sort by Net Expected High to Low
    teams.sort(key=lambda x: x["net_expected"], reverse=True)
    return teams

@app.get("/api/team/{team_id}")
@cache_response(expire_seconds=3600)
def get_team(team_id: int):
    if not predictor.models_loaded:
        raise HTTPException(status_code=503, detail="Modelos no cargados")
    try:
        s = predictor.get_team_stats(team_id)
        WEST = {1610612742, 1610612743, 1610612744, 1610612745, 1610612746, 1610612747, 1610612763, 1610612750, 1610612740, 1610612760, 1610612756, 1610612757, 1610612758, 1610612759, 1610612762}
        s["conf"] = "West" if team_id in WEST else "East"
        return s
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/api/predict")
def predict(req: PredictionRequest, background_tasks: BackgroundTasks, db = Depends(get_db)):
    if not predictor.models_loaded:
        raise HTTPException(status_code=503, detail="Modelos no cargados")
    
    try:
        result = predictor.predict(req.team1, req.team2, home_team=req.home_team)
        # Background audit logging
        background_tasks.add_task(save_prediction_db, db, req.team1, req.team2, req.home_team, result)
        
        logger.info(f"POST /api/predict -> {req.team1} vs {req.team2} = {result['prediction']}")
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Predicción falló: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@app.post("/api/head-to-head")
def head_to_head(req: HeadToHeadRequest):
    if not predictor.models_loaded:
        raise HTTPException(status_code=503, detail="Modelos no cargados")
    
    try:
        t1 = predictor.get_team_stats(req.team1)
        t2 = predictor.get_team_stats(req.team2)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
        
    random.seed(hash(str(req.team1) + str(req.team2)))
    games = []
    t1_wins = 0
    t2_wins = 0
    
    for i in range(5):
        # Aumentamos la varianza (Desviación Típica de 8 puntos) para que la probabilidad estadística
        # permita resultados sorpresas en los 5 juegos.
        s1 = int(round(random.gauss(t1["expectedTeamScore"], 9)))
        s2 = int(round(random.gauss(t2["expectedTeamScore"], 9)))
        
        # Eliminar empates
        if s1 == s2:
            s1 += 1
            
        winner_id = req.team1 if s1 > s2 else req.team2
        winner_name = t1["team_name"] if s1 > s2 else t2["team_name"]
        
        if s1 > s2:
            t1_wins += 1
        else:
            t2_wins += 1
            
        games.append({
            "game": i + 1,
            "t1_score": s1,
            "t2_score": s2,
            "winner_id": winner_id,
            "winner_name": winner_name
        })
        
    return {
        "team1_id": req.team1,
        "team2_id": req.team2,
        "team1_name": t1["team_name"],
        "team2_name": t2["team_name"],
        "team1_wins": t1_wins,
        "team2_wins": t2_wins,
        "series_winner": t1["team_name"] if t1_wins > t2_wins else t2["team_name"],
        "games": games
    }

@app.get("/api/model/info")
@cache_response(expire_seconds=3600)
def model_info():
    if not predictor.models_loaded:
        raise HTTPException(status_code=503, detail="Modelos no cargados")
        
    feat_imp = predictor.clf_importances
    top_feats = sorted(
        [{"feature": f, "importance": round(float(i), 2)}
         for f, i in zip(predictor.clf_features, feat_imp)],
        key=lambda x: -x["importance"]
    )
    return {
        "model_type": "Calibrated VotingClassifier (Ensemble)",
        "model_version": "2.0.0",
        "num_features": len(predictor.clf_features),
        "top_10_features": top_feats[:10],
        "teams_in_snapshot": len(predictor.snapshot),
    }

@app.get("/api/metrics")
def get_metrics(limit: int = 50, db = Depends(get_db)):
    """Métricas de aciertos en la DB y logs recientes."""
    hr = calculate_hit_rate(db)
    logs = get_last_predictions(db, limit)
    
    history = []
    for l in logs:
        history.append({
            "id": l.id,
            "team1": TEAM_NAMES.get(l.team1_id, str(l.team1_id)),
            "team2": TEAM_NAMES.get(l.team2_id, str(l.team2_id)),
            "home": l.home_team,
            "predicted": l.predicted_winner,
            "prob": l.win_probability,
            "actual_winner": l.actual_winner,
            "correct": l.is_correct,
            "date": l.created_at.isoformat()
        })
        
    return {
        "metrics": hr,
        "history": history
    }

@app.post("/api/admin/update")
def admin_update(req: AdminUpdateRequest, background_tasks: BackgroundTasks):
    if req.secret != ADMIN_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")
        
    background_tasks.add_task(run_update_pipeline)
    return {"status": "Update pipeline strated in background"}

