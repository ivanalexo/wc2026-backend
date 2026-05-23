from datetime import datetime

from sqlalchemy import String, Float, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base

class Team(Base):
    __tablename__ = 'teams'

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, index=True, nullable=False)
    slug: Mapped[str] = mapped_column(String(100), unique=True, index=True, nullable=False)
    flag_code: Mapped[str] = mapped_column(String(3), nullable=True)  # Código ISO del país para la bandera
    group: Mapped[str | None] = mapped_column(String(1))
    confederation: Mapped[str | None] = mapped_column(String(10))
    elo_rating: Mapped[float | None] = mapped_column(Float)

    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    def __repr__(self):
        return f'<Team {self.name} | ELO={self.elo_rating}>'