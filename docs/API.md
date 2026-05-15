# API Reference

The NBA ML Predictor backend provides a FastAPI-基于 REST API for interacting with the predictive ensemble models and viewing team standings.

Base URL: `http://localhost:8000/api`

## Endpoints Completos

### 1. `GET /health`
Returns the status of the API, the Redis cluster connection, and the model loading state.
**Caché:** None.

### 2. `GET /teams`
Returns all NBA teams with their expected scores, relative strength, and form.
**Caché:** 1 Hour (Fault-Tolerant).

### 3. `GET /standings?conf={East|West}`
Returns all NBA teams sorted by their `net_expected` strength score. Can be filtered by conference.
**Caché:** 1 Hour (Fault-Tolerant).

### 4. `GET /team/{team_id}`
Returns granular statistics for a single NBA team.
**Caché:** 1 Hour (Fault-Tolerant).

### 5. `POST /predict`
Runs the ML Ensemble (CatBoost + LightGBM + Ridge) on a given matchup.
**Parameters (JSON Body):**
```json
{
  "team1": 1610612747,
  "team2": 1610612738,
  "home_team": "team1"
}
```
**Returns:** Probabilities of winning.

### 6. `POST /head-to-head`
Simulates 5 consecutive games adding gaussian noise to Expected Scores to determine series outcomes.
**Parameters (JSON Body):**
```json
{
  "team1": 1610612747,
  "team2": 1610612738
}
```

### 7. `GET /metrics`
Returns the hit-rate of the API (how many predictions were actually correct) alongside the history.

### 8. `GET /model/info`
Returns information on the active ML Ensemble and its Top Features.
