from datetime import datetime

from sqlalchemy import String, Float, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class SimulationResult(Base):
    """
    Probabilidades por equipo de la última simulación Monte Carlo.

    Es la fuente de verdad que sirve /simulate (reemplaza al CSV
    montecarlo_probabilities.csv). Vive en Postgres para sobrevivir redeploys
    de Railway y ser consistente entre múltiples workers.
    """
    __tablename__ = "simulation_results"

    team: Mapped[str] = mapped_column(String(100), primary_key=True)

    elo: Mapped[float | None] = mapped_column(Float)

    p_qualify: Mapped[float] = mapped_column(Float)
    p_reach_r16: Mapped[float] = mapped_column(Float)
    p_reach_qf: Mapped[float] = mapped_column(Float)
    p_reach_sf: Mapped[float] = mapped_column(Float)
    p_reach_final: Mapped[float] = mapped_column(Float)
    p_champion: Mapped[float] = mapped_column(Float)

    updated_at: Mapped[datetime] = mapped_column(
        server_default=func.now(), onupdate=func.now()
    )

    def __repr__(self) -> str:
        return f"<SimulationResult {self.team} | champ={self.p_champion:.3f}>"
