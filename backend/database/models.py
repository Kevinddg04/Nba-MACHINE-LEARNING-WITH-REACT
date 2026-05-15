"""
database/models.py
==================
SQLAlchemy models for prediction auditing and tracking.
"""

from sqlalchemy import Column, Integer, Float, String, DateTime, Boolean
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()


class PredictionLog(Base):
    """
    Stores every prediction made by the API for auditing.
    
    Fields actual_winner_id, actual_total_points, and is_correct
    are filled in after the game is played (via admin endpoint or script).
    """

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    # Request data
    team1_id = Column(Integer, nullable=False)
    team2_id = Column(Integer, nullable=False)
    home_team = Column(String(10), nullable=False)

    # Prediction output
    predicted_winner_id = Column(Integer, nullable=False)
    predicted_winner_prob = Column(Float, nullable=False)  # 0.0 - 1.0
    predicted_total_points = Column(Integer, nullable=True)

    # Actual results (filled later)
    actual_winner_id = Column(Integer, nullable=True)
    actual_total_points = Column(Integer, nullable=True)
    is_correct = Column(Boolean, nullable=True)

    # Model version tracking
    model_version = Column(String(50), default="v1.0")

    def __repr__(self):
        return (
            f"<PredictionLog {self.id}: "
            f"T{self.team1_id} vs T{self.team2_id} "
            f"→ T{self.predicted_winner_id} ({self.predicted_winner_prob:.1%})>"
        )

    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "team1_id": self.team1_id,
            "team2_id": self.team2_id,
            "home_team": self.home_team,
            "predicted_winner_id": self.predicted_winner_id,
            "predicted_winner_prob": round(self.predicted_winner_prob, 4),
            "predicted_total_points": self.predicted_total_points,
            "actual_winner_id": self.actual_winner_id,
            "actual_total_points": self.actual_total_points,
            "is_correct": self.is_correct,
            "model_version": self.model_version,
        }
