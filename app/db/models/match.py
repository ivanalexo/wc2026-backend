from datetime import datetime

from sqlalchemy import String, Integer, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base

class Match(Base):
    __tablename__ = 'matches'

    id: Mapped[int] = mapped_column(primary_key=True)

    home_team: Mapped[str] = mapped_column(String(100), index=True)
    away_team: Mapped[str] = mapped_column(String(100), index=True)

    date: Mapped[datetime] = mapped_column(index=True)

    city: Mapped[str | None] = mapped_column(String(100))
    country: Mapped[str | None] = mapped_column(String(100))

    # Fase del torneo
    stage: Mapped[str | None] = mapped_column(String(50))

    # Grupo
    group: Mapped[str | None] = mapped_column(String(1))

    # Estado del partido: scheduled | live | finished
    status: Mapped[str] = mapped_column(String(20), default="scheduled", server_default="scheduled")

    # Resultados
    home_score: Mapped[int | None] = mapped_column(Integer)
    away_score: Mapped[int | None] = mapped_column(Integer)

    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    def __repr__(self) -> str:
        return f"<Match {self.home_team} vs {self.away_team} | {self.date.date()}>"
