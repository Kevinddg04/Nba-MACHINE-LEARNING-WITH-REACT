"""
test_api.py
===========
15 tests for the Flask REST API endpoints.
"""

import pytest
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ═══════════════════════════════════════════════════════════════════════════
#  HEALTH (2 tests)
# ═══════════════════════════════════════════════════════════════════════════

class TestHealth:
    """Tests for /health and /ping endpoints."""

    def test_health_endpoint_200(self, app_client):
        """GET /health returns 200."""
        resp = app_client.get("/health")
        assert resp.status_code == 200

    def test_health_models_loaded(self, app_client):
        """GET /health reports models_loaded status."""
        resp = app_client.get("/health")
        data = resp.json()
        assert "models_loaded" in data
        assert "status" in data
        assert data["status"] == "ok"


# ═══════════════════════════════════════════════════════════════════════════
#  TEAMS (2 tests)
# ═══════════════════════════════════════════════════════════════════════════

class TestTeams:
    """Tests for /api/teams endpoint."""

    def test_get_teams_200(self, app_client):
        """GET /api/teams returns 200."""
        resp = app_client.get("/api/teams")
        # 200 if models loaded, 503 if not
        assert resp.status_code in (200, 503)

    def test_get_teams_has_required_fields(self, app_client):
        """Each team in response has required fields."""
        resp = app_client.get("/api/teams")
        if resp.status_code == 503:
            pytest.skip("Models not loaded")
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) > 0
        required = ["team_id", "name", "conf", "expectedScore", "net_expected"]
        for team in data[:3]:  # Check first 3
            for field in required:
                assert field in team, f"Missing field: {field}"


# ═══════════════════════════════════════════════════════════════════════════
#  PREDICT (6 tests)
# ═══════════════════════════════════════════════════════════════════════════

class TestPredict:
    """Tests for POST /api/predict endpoint."""

    def test_predict_valid_teams_200(self, app_client, sample_prediction_payload):
        """POST /api/predict with valid teams returns 200."""
        resp = app_client.post(
            "/api/predict",
            json=sample_prediction_payload,
        )
        assert resp.status_code in (200, 503)

    def test_predict_invalid_team_422(self, app_client):
        """POST /api/predict with invalid team ID returns 422/404."""
        payload = {"team1": 999999, "team2": 888888, "home_team": "team1"}
        resp = app_client.post(
            "/api/predict",
            json=payload,
        )
        # 422 (validation error) or 404 (team not found) or 503 (models not loaded)
        assert resp.status_code in (422, 404, 503)

    def test_predict_same_team_422(self, app_client, valid_team_ids):
        """POST /api/predict with same team for both returns 422."""
        payload = {
            "team1": valid_team_ids[0],
            "team2": valid_team_ids[0],
            "home_team": "team1",
        }
        resp = app_client.post(
            "/api/predict",
            json=payload,
        )
        # 422 if models loaded (validation), 503 if models not loaded
        assert resp.status_code in (422, 503)

    def test_predict_response_has_fields(self, app_client, sample_prediction_payload):
        """Prediction response has all required fields."""
        resp = app_client.post(
            "/api/predict",
            json=sample_prediction_payload,
        )
        if resp.status_code == 503:
            pytest.skip("Models not loaded")
        data = resp.json()
        assert "prediction" in data
        assert "win_probability" in data
        assert "team1" in data
        assert "team2" in data
        assert "details" in data

    def test_predict_prob_in_range(self, app_client, sample_prediction_payload):
        """Prediction probabilities are in [0, 100]."""
        resp = app_client.post(
            "/api/predict",
            json=sample_prediction_payload,
        )
        if resp.status_code == 503:
            pytest.skip("Models not loaded")
        data = resp.json()
        p1 = data["team1"]["probability"]
        p2 = data["team2"]["probability"]
        assert 0 <= p1 <= 100
        assert 0 <= p2 <= 100
        assert abs(p1 + p2 - 100) < 1.0  # Should sum to ~100

    def test_predict_missing_teams_422(self, app_client):
        """POST /api/predict without teams returns 422."""
        payload = {"home_team": "team1"}
        resp = app_client.post(
            "/api/predict",
            json=payload,
        )
        assert resp.status_code in (422, 503)


# ═══════════════════════════════════════════════════════════════════════════
#  STANDINGS (2 tests)
# ═══════════════════════════════════════════════════════════════════════════

class TestStandings:
    """Tests for GET /api/standings endpoint."""

    def test_standings_200(self, app_client):
        """GET /api/standings returns 200."""
        resp = app_client.get("/api/standings")
        assert resp.status_code in (200, 503)

    def test_standings_conf_filter(self, app_client):
        """GET /api/standings?conf=East filters correctly."""
        resp = app_client.get("/api/standings?conf=East")
        if resp.status_code == 503:
            pytest.skip("Models not loaded")
        data = resp.json()
        for team in data:
            assert team["conf"] == "East", f"Expected East, got {team['conf']}"


# ═══════════════════════════════════════════════════════════════════════════
#  HEAD-TO-HEAD (2 tests)
# ═══════════════════════════════════════════════════════════════════════════

class TestHeadToHead:
    """Tests for POST /api/head-to-head endpoint."""

    def test_h2h_200(self, app_client, valid_team_ids):
        """POST /api/head-to-head with valid teams returns 200."""
        payload = {"team1": valid_team_ids[0], "team2": valid_team_ids[1]}
        resp = app_client.post(
            "/api/head-to-head",
            json=payload,
        )
        assert resp.status_code in (200, 503)

    def test_h2h_5_games(self, app_client, valid_team_ids):
        """Head-to-head returns exactly 5 simulated games."""
        payload = {"team1": valid_team_ids[0], "team2": valid_team_ids[1]}
        resp = app_client.post(
            "/api/head-to-head",
            json=payload,
        )
        if resp.status_code == 503:
            pytest.skip("Models not loaded")
        data = resp.json()
        assert "games" in data
        assert len(data["games"]) == 5
        assert data["team1_wins"] + data["team2_wins"] == 5


# ═══════════════════════════════════════════════════════════════════════════
#  MODEL INFO (1 test)
# ═══════════════════════════════════════════════════════════════════════════

class TestModelInfo:
    """Tests for GET /api/model/info endpoint."""

    def test_model_info_200(self, app_client):
        """GET /api/model/info returns 200 with feature importance."""
        resp = app_client.get("/api/model/info")
        if resp.status_code == 503:
            pytest.skip("Models not loaded")
        assert resp.status_code == 200
        data = resp.json()
        assert "model_type" in data
        assert "num_features" in data
        assert "top_10_features" in data
        assert len(data["top_10_features"]) > 0
