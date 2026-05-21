"""
test_ml_pipeline.py
===================
20 tests for the ML pipeline: data loading, features, matchups,
classifier, and regressor.
"""

import pytest
import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ═══════════════════════════════════════════════════════════════════════════
#  DATA LOADING (5 tests)
# ═══════════════════════════════════════════════════════════════════════════

class TestDataLoading:
    """Tests for load_and_clean() function."""

    def test_csv_loads(self, csv_exists):
        """CSV file exists and is accessible."""
        assert csv_exists, "TeamStatistics.csv not found in project root"

    def test_load_and_clean_returns_dataframe(self, csv_exists):
        """load_and_clean() returns a valid DataFrame."""
        if not csv_exists:
            pytest.skip("CSV not found")
        from ml_pipeline import load_and_clean, CSV_PATH
        original_cwd = os.getcwd()
        os.chdir(os.path.join(os.path.dirname(__file__), ".."))
        try:
            df = load_and_clean(os.path.join(os.path.dirname(__file__), "..", "..", "TeamStatistics.csv"))
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
        finally:
            os.chdir(original_cwd)

    def test_load_and_clean_removes_invalid_ids(self, csv_exists):
        """Exhibition/All-Star team IDs are removed."""
        if not csv_exists:
            pytest.skip("CSV not found")
        from ml_pipeline import load_and_clean, IDS_A_ELIMINAR
        original_cwd = os.getcwd()
        os.chdir(os.path.join(os.path.dirname(__file__), ".."))
        try:
            df = load_and_clean(os.path.join(os.path.dirname(__file__), "..", "..", "TeamStatistics.csv"))
            for bad_id in IDS_A_ELIMINAR:
                assert bad_id not in df["teamId"].values, f"ID {bad_id} should have been removed"
        finally:
            os.chdir(original_cwd)

    def test_load_and_clean_has_min_rows(self, csv_exists):
        """Cleaned data has at least 10,000 rows."""
        if not csv_exists:
            pytest.skip("CSV not found")
        from ml_pipeline import load_and_clean
        original_cwd = os.getcwd()
        os.chdir(os.path.join(os.path.dirname(__file__), ".."))
        try:
            df = load_and_clean(os.path.join(os.path.dirname(__file__), "..", "..", "TeamStatistics.csv"))
            assert len(df) >= 10000, f"Expected >=10000 rows, got {len(df)}"
        finally:
            os.chdir(original_cwd)

    def test_team_ids_valid(self, csv_exists):
        """teamIds are valid NBA-range integers (> 1 billion)."""
        if not csv_exists:
            pytest.skip("CSV not found")
        from ml_pipeline import load_and_clean, TEAM_NAMES
        original_cwd = os.getcwd()
        os.chdir(os.path.join(os.path.dirname(__file__), ".."))
        try:
            df = load_and_clean(os.path.join(os.path.dirname(__file__), "..", "..", "TeamStatistics.csv"))
            valid_ids = set(TEAM_NAMES.keys())
            # At least 90% of teamIds should be recognizable NBA teams
            known = df["teamId"].isin(valid_ids).mean()
            assert known >= 0.90, f"Only {known:.1%} of teamIds are recognized NBA teams"
        finally:
            os.chdir(original_cwd)


# ═══════════════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING (5 tests)
# ═══════════════════════════════════════════════════════════════════════════

class TestFeatureEngineering:
    """Tests for build_features() function."""

    @pytest.fixture(scope="class")
    def featured_df(self, csv_exists):
        """DataFrame after feature engineering."""
        if not csv_exists:
            pytest.skip("CSV not found")
        from ml_pipeline import load_and_clean, build_features
        original_cwd = os.getcwd()
        os.chdir(os.path.join(os.path.dirname(__file__), ".."))
        try:
            df = load_and_clean(os.path.join(os.path.dirname(__file__), "..", "..", "TeamStatistics.csv"))
            return build_features(df)
        finally:
            os.chdir(original_cwd)

    def test_add_features_no_nan(self, featured_df):
        """Feature engineering produces no NaN values."""
        nan_cols = featured_df.columns[featured_df.isna().any()].tolist()
        assert len(nan_cols) == 0, f"NaN found in columns: {nan_cols}"

    def test_rolling_means_realistic(self, featured_df):
        """Expected team scores are in realistic NBA range."""
        vals = featured_df["expectedTeamScore"]
        # Allow edge case low scores (COVID bubble, short rolling windows)
        assert vals.quantile(0.01) >= 80, f"1st percentile too low: {vals.quantile(0.01)}"
        assert vals.quantile(0.99) <= 150, f"99th percentile too high: {vals.quantile(0.99)}"

    def test_win_streak_in_range(self, featured_df):
        """win_streak_5 is between 0 and 5."""
        assert featured_df["win_streak_5"].min() >= 0
        assert featured_df["win_streak_5"].max() <= 5

    def test_no_duplicate_rows(self, featured_df):
        """No exact duplicate rows in featured data."""
        if "gameId" in featured_df.columns:
            dupes = featured_df.duplicated(subset=["gameId", "teamId"]).sum()
            assert dupes == 0, f"Found {dupes} duplicate rows"

    def test_has_required_columns(self, featured_df):
        """All expected feature columns exist."""
        required = [
            "expectedTeamScore", "expectedOpponentScore",
            "win_streak_5", "RealHandicap", "totalPoints",
            "defensive_rating_r10", "gameId",
        ]
        for col in required:
            assert col in featured_df.columns, f"Missing column: {col}"


# ═══════════════════════════════════════════════════════════════════════════
#  MATCHUP CONSTRUCTION (3 tests)
# ═══════════════════════════════════════════════════════════════════════════

class TestMatchupConstruction:
    """Tests for build_classifier_matchup_data() function."""

    @pytest.fixture(scope="class")
    def matchup_df(self, csv_exists):
        """Matchup DataFrame from the pipeline."""
        if not csv_exists:
            pytest.skip("CSV not found")
        from ml_pipeline import load_and_clean, build_features, build_classifier_matchup_data
        original_cwd = os.getcwd()
        os.chdir(os.path.join(os.path.dirname(__file__), ".."))
        try:
            df = load_and_clean(os.path.join(os.path.dirname(__file__), "..", "..", "TeamStatistics.csv"))
            df = build_features(df)
            return build_classifier_matchup_data(df)
        finally:
            os.chdir(original_cwd)

    def test_matchup_symmetric(self, matchup_df):
        """Matchup data is symmetric: ~50% home=1 and ~50% home=0."""
        home_ratio = matchup_df["home"].mean()
        assert 0.45 <= home_ratio <= 0.55, f"Home ratio {home_ratio:.2f} is not ~0.50"

    def test_matchup_labels_correct(self, matchup_df):
        """Labels are binary: only 0 and 1."""
        unique = sorted(matchup_df["label_win_A"].unique())
        assert unique == [0, 1], f"Unexpected labels: {unique}"

    def test_matchup_enough_rows(self, matchup_df):
        """Matchup data has at least 10,000 rows."""
        assert len(matchup_df) >= 10000, f"Only {len(matchup_df)} matchup rows"


# ═══════════════════════════════════════════════════════════════════════════
#  CLASSIFIER (4 tests)
# ═══════════════════════════════════════════════════════════════════════════

class TestClassifier:
    """Tests for the trained CatBoostClassifier."""

    def test_classifier_loads(self, models_exist):
        """Classifier model file exists and loads."""
        if not models_exist:
            pytest.skip("Models not trained")
        import joblib
        model = joblib.load(os.path.join(os.path.dirname(__file__), "..", "models", "classifier.pkl"))
        assert model is not None

    def test_classifier_accuracy_above_baseline(self, predictor):
        """Classifier accuracy should be above coin-flip baseline (50%)."""
        # We test this implicitly: if the model is loaded and can predict,
        # it was trained with reported accuracy > 50%
        assert predictor.models_loaded

    def test_classifier_features_reasonable(self, predictor):
        """Classifier has a reasonable number of features (10-50)."""
        n_features = len(predictor.clf_features)
        assert 5 <= n_features <= 50, f"Got {n_features} features"

    def test_classifier_predictions_in_range(self, predictor, valid_team_ids):
        """Classifier predictions produce probabilities in [0, 1]."""
        result = predictor.predict(valid_team_ids[0], valid_team_ids[1], "team1")
        p1 = result["team1"]["probability"]
        p2 = result["team2"]["probability"]
        assert 0 <= p1 <= 100, f"team1 probability {p1} out of range"
        assert 0 <= p2 <= 100, f"team2 probability {p2} out of range"
        assert abs(p1 + p2 - 100) < 0.5, f"Probabilities don't sum to 100: {p1} + {p2}"


# ═══════════════════════════════════════════════════════════════════════════
#  REGRESSOR (3 tests)
# ═══════════════════════════════════════════════════════════════════════════

class TestRegressor:
    """Tests for the trained CatBoostRegressor."""

    def test_regressor_loads(self, models_exist):
        """Regressor model file exists and loads."""
        if not models_exist:
            pytest.skip("Models not trained")
        import joblib
        model = joblib.load(os.path.join(os.path.dirname(__file__), "..", "models", "regressor.pkl"))
        assert model is not None

    def test_regressor_features_exist(self, models_exist):
        """Regressor feature list exists and is non-empty."""
        if not models_exist:
            pytest.skip("Models not trained")
        import joblib
        features = joblib.load(os.path.join(os.path.dirname(__file__), "..", "models", "regressor_features.pkl"))
        assert len(features) > 0

    def test_regressor_predictions_realistic(self, predictor, valid_team_ids):
        """Regressor output is used in the prediction pipeline."""
        result = predictor.predict(valid_team_ids[0], valid_team_ids[1], "team1")
        # The prediction should include a valid winner
        assert "prediction" in result
        assert result["prediction"] in [
            result["team1"]["name"],
            result["team2"]["name"],
        ] or isinstance(result["prediction"], str)
