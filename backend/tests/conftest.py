"""
conftest.py
===========
Shared pytest fixtures for NBA ML Predictor test suite.
"""

import pytest
import sys
import os
import pandas as pd

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
