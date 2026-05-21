"""
conftest.py
===========
Fixtures compartidos para la suite de tests del predictor NBA.

Nota CI: Si CatBoost o LightGBM no están instalados (entorno de GitHub Actions),
se inyectan stubs automáticos en sys.modules antes de cualquier importación.
Esto garantiza que los tests de API funcionen sin esas librerías pesadas.
"""

import pytest
import sys
import os
import types
import pandas as pd

# ── Stubs para entornos CI sin CatBoost/LightGBM ──────────────────────────
def _install_ml_stubs():
    """Instala mocks mínimos de CatBoost y LightGBM si no están disponibles."""
    try:
        import catboost  # noqa: F401
    except ImportError:
        import numpy as _np

        cb = types.ModuleType("catboost")

        class _CB:
            def __init__(self, **kw): self.feature_importances_ = _np.array([1.0])
            def fit(self, *a, **kw): return self
            def predict(self, X): return [0] * len(X)
            def predict_proba(self, X): return [[0.5, 0.5]] * len(X)
            def get_feature_importance(self): return _np.array([1.0])

        class _CBR:
            def __init__(self, **kw): pass
            def fit(self, *a, **kw): return self
            def predict(self, X): return [110.0] * len(X)

        cb.CatBoostClassifier = _CB
        cb.CatBoostRegressor = _CBR
        sys.modules["catboost"] = cb

    try:
        import lightgbm  # noqa: F401
    except ImportError:
        lgb = types.ModuleType("lightgbm")

        class _LGBM:
            def __init__(self, **kw): pass
            def fit(self, *a, **kw): return self
            def predict(self, X): return [0] * len(X)
            def predict_proba(self, X): return [[0.5, 0.5]] * len(X)

        lgb.LGBMClassifier = _LGBM
        sys.modules["lightgbm"] = lgb

    try:
        import statsmodels  # noqa: F401
    except ImportError:
        sm = types.ModuleType("statsmodels")
        sm_stats = types.ModuleType("statsmodels.stats")
        sm_outliers = types.ModuleType("statsmodels.stats.outliers_influence")
        sm_outliers.variance_inflation_factor = lambda X, i: 1.0
        sys.modules["statsmodels"] = sm
        sys.modules["statsmodels.stats"] = sm_stats
        sys.modules["statsmodels.stats.outliers_influence"] = sm_outliers


_install_ml_stubs()

# Add backend directory to path so we can import modules directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Paths ──────────────────────────────────────────────────────────────────

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)
CSV_PATH = os.path.join(PROJECT_ROOT, "TeamStatistics.csv")
MODELS_DIR = os.path.join(BACKEND_DIR, "models")


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def csv_path():
    """Path to the main CSV dataset."""
    return CSV_PATH


@pytest.fixture(scope="session")
def csv_exists():
    """Check if the CSV file exists."""
    return os.path.exists(CSV_PATH)


@pytest.fixture(scope="session")
def raw_df():
    """Load a subset of the CSV for fast testing (first 5000 rows)."""
    if not os.path.exists(CSV_PATH):
        pytest.skip("TeamStatistics.csv not found — skipping data tests")
    df = pd.read_csv(CSV_PATH, index_col=0, low_memory=False, nrows=10000)
    return df


@pytest.fixture(scope="session")
def full_df():
    """Load the complete CSV (slow — use only for marked tests)."""
    if not os.path.exists(CSV_PATH):
        pytest.skip("TeamStatistics.csv not found")
    return pd.read_csv(CSV_PATH, index_col=0, low_memory=False)


@pytest.fixture(scope="session")
def app_client():
    """Flask test client with the real app."""
    # Change working directory to backend so model paths resolve correctly
    original_cwd = os.getcwd()
    os.chdir(BACKEND_DIR)

    try:
        from main import app
        from fastapi.testclient import TestClient
        with TestClient(app) as client:
            yield client
    finally:
        os.chdir(original_cwd)


@pytest.fixture(scope="session")
def predictor():
    """NBAPredictor instance with loaded models."""
    original_cwd = os.getcwd()
    os.chdir(BACKEND_DIR)

    try:
        from ml_pipeline import NBAPredictor
        p = NBAPredictor()
        if not p.models_loaded:
            pytest.skip("Models not trained — run 'python ml_pipeline.py' first")
        yield p
    finally:
        os.chdir(original_cwd)


@pytest.fixture(scope="session")
def models_exist():
    """Check if trained models exist."""
    required = [
        "classifier.pkl",
        "regressor.pkl",
        "classifier_features.pkl",
        "regressor_features.pkl",
        "team_stats_snapshot.pkl",
    ]
    return all(os.path.exists(os.path.join(MODELS_DIR, f)) for f in required)


@pytest.fixture(scope="session")
def team_names():
    """Dictionary of valid team IDs → names."""
    from ml_pipeline import TEAM_NAMES
    return TEAM_NAMES


@pytest.fixture(scope="session")
def valid_team_ids(team_names):
    """List of valid NBA team IDs."""
    return list(team_names.keys())


@pytest.fixture
def sample_prediction_payload(valid_team_ids):
    """Sample valid prediction request body."""
    return {
        "team1": valid_team_ids[0],
        "team2": valid_team_ids[1],
        "home_team": "team1",
    }
