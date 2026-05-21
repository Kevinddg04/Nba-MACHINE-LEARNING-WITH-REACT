"""
metrics/calculator.py
=====================
Calculates prediction hit rate and retrieves prediction history.
Uses the PredictionLog table for auditing model performance.
"""

from database import SessionLocal
from database.models import PredictionLog
from datetime import datetime, timedelta
from sqlalchemy import func


def calculate_hit_rate(days_back=None):
    """
    Calculates the hit rate (accuracy) of predictions.

    Args:
        days_back: Number of days to look back. None = all time.

    Returns:
        dict with total, correct, and hit_rate
    """
    db = SessionLocal()
    try:
        query = db.query(PredictionLog).filter(
            PredictionLog.is_correct.isnot(None)
        )

        if days_back:
            start_date = datetime.utcnow() - timedelta(days=days_back)
            query = query.filter(PredictionLog.timestamp >= start_date)

        total = query.count()
        correct = query.filter(PredictionLog.is_correct == True).count()

        return {
            "total": total,
            "correct": correct,
            "hit_rate": round(correct / total, 4) if total > 0 else 0,
        }
    finally:
        db.close()


def get_last_predictions(limit=10):
    """
    Retrieves the last N predictions with their results.

    Args:
        limit: Number of predictions to retrieve.

    Returns:
        List of prediction dicts
    """
    db = SessionLocal()
    try:
        predictions = (
            db.query(PredictionLog)
            .order_by(PredictionLog.timestamp.desc())
            .limit(limit)
            .all()
        )

        return [
            {
                "id": p.id,
                "date": p.timestamp.isoformat() if p.timestamp else None,
                "team1_id": p.team1_id,
                "team2_id": p.team2_id,
                "home_team": p.home_team,
                "predicted_winner": p.predicted_winner_id,
                "predicted_prob": round(p.predicted_winner_prob, 4),
                "actual_winner": p.actual_winner_id,
                "correct": p.is_correct,
            }
            for p in predictions
        ]
    finally:
        db.close()


def get_prediction_count():
    """Returns total number of logged predictions."""
    db = SessionLocal()
    try:
        return db.query(func.count(PredictionLog.id)).scalar()
    finally:
        db.close()
