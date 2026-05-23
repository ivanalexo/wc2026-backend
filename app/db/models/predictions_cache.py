from datetime import datetime

from sqlalchemy import String, Float, JSON, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base

class PredictionsCache(Base):
    __tablename__ = 'predictions_cache'

    # Un solo registro por par de equipos, si se vuelve a predcir se actualiza
    __table_args__ = (
        UniqueConstraint('home_team', 'away_team', name='uq_prediction_pair'),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    home_team: Mapped[str] = mapped_column(String(100), index=True)
    away_team: Mapped[str] = mapped_column(String(100), index=True)

    # Probabilidades del modelo XGBoost
    p_home_win: Mapped[float] = mapped_column(Float)
    p_draw: Mapped[float] = mapped_column(Float)
    p_away_win: Mapped[float] = mapped_column(Float)

    # Outcome mas probable 'H', 'D' o 'A'
    prediction: Mapped[str] = mapped_column(String(1))

    # Vector de features de entrada (19 valore numericos)
    features: Mapped[dict] = mapped_column(JSON)

    # Frases generada a parir de SHAP values
    shap_explanation: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    def __repr__(self) -> str:
        return (
            f"<Prediction {self.home_team} vs {self.away_team} "
            f"| {self.prediction} ({self.p_home_win:.0%}/{self.p_draw:.0%}/{self.p_away_win:.0%})>"
        )