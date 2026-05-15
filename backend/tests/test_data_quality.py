"""
test_data_quality.py
====================
8 tests for validating the quality of TeamStatistics.csv data.
"""

import pytest
import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ═══════════════════════════════════════════════════════════════════════════
#  DATA QUALITY (8 tests)
# ═══════════════════════════════════════════════════════════════════════════


def test_csv_columns_exist(raw_df):
    """CSV has the essential columns for the pipeline."""
    required_columns = [
        "teamId", "opponentTeamId", "gameDateTimeEst",
        "teamScore", "opponentScore", "win", "home",
    ]
    for col in required_columns:
        assert col in raw_df.columns, f"Missing essential column: {col}"


def test_no_duplicate_game_rows(raw_df):
    """No duplicate (gameId, teamId) pairs if gameId column exists."""
    if "gameId" not in raw_df.columns:
        pytest.skip("gameId column not present in raw data")
    dupes = raw_df.duplicated(subset=["gameId", "teamId"]).sum()
    # Allow small number of dupes in raw data (cleaned later)
    assert dupes < len(raw_df) * 0.01, f"Too many duplicates: {dupes}"


def test_team_ids_range(raw_df):
    """Team IDs are in expected NBA range (1610612737 - 1610612766)."""
    from ml_pipeline import TEAM_NAMES
    valid_ids = set(TEAM_NAMES.keys())

    # Filter out exhibition/all-star IDs
    from ml_pipeline import IDS_A_ELIMINAR
    team_ids = set(raw_df["teamId"].unique()) - set(IDS_A_ELIMINAR)

    # At least 25 of the 30 NBA teams should be present
    nba_teams_present = team_ids & valid_ids
    assert len(nba_teams_present) >= 25, (
        f"Only {len(nba_teams_present)} NBA teams found, expected >= 25"
    )


def test_scores_realistic(raw_df):
    """Team scores are in realistic NBA range for normal games."""
    scores = raw_df["teamScore"].dropna()
    # Filter out forfeits/exhibitions with 0 scores
    normal_scores = scores[scores > 0]
    assert len(normal_scores) > len(scores) * 0.95, "Too many zero-score games"
    assert normal_scores.max() <= 200, f"Max score too high: {normal_scores.max()}"

    mean_score = normal_scores.mean()
    assert 90 <= mean_score <= 130, (
        f"Mean score {mean_score:.1f} outside expected range [90, 130]"
    )


def test_dates_chronological(raw_df):
    """Dates are parseable and span multiple seasons."""
    dates = pd.to_datetime(
        raw_df["gameDateTimeEst"], errors="coerce", format="mixed", utc=True
    ).dropna()

    assert len(dates) > 0, "No valid dates found"

    min_year = dates.min().year
    max_year = dates.max().year
    assert max_year - min_year >= 3, (
        f"Date range too narrow: {min_year}-{max_year}"
    )


def test_no_future_dates(raw_df):
    """No game dates are unreasonably far in the future."""
    dates = pd.to_datetime(
        raw_df["gameDateTimeEst"], errors="coerce", format="mixed", utc=True
    ).dropna()

    # Allow up to end of 2027 (generous buffer)
    max_allowed = pd.Timestamp("2028-01-01", tz="UTC")
    future_games = (dates > max_allowed).sum()
    assert future_games == 0, f"Found {future_games} games after 2028"


def test_balanced_home_away(raw_df):
    """Home/Away split is roughly balanced (~50/50)."""
    if "home" not in raw_df.columns:
        pytest.skip("No 'home' column")
    home_ratio = raw_df["home"].mean()
    assert 0.40 <= home_ratio <= 0.60, (
        f"Home ratio {home_ratio:.2f} is too imbalanced"
    )


def test_team_stats_snapshot_valid(models_exist):
    """Team stats snapshot has 30 teams with valid data."""
    if not models_exist:
        pytest.skip("Models not trained")
    import joblib
    snapshot = joblib.load(
        os.path.join(os.path.dirname(__file__), "..", "models", "team_stats_snapshot.pkl")
    )

    assert isinstance(snapshot, pd.DataFrame)
    assert len(snapshot) >= 28, f"Only {len(snapshot)} teams in snapshot (expected >= 28)"
    assert "teamId" in snapshot.columns
    assert "expectedTeamScore" in snapshot.columns
