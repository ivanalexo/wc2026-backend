from datetime import datetime

from sqlalchemy import String, Integer, Float, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base

class Player(Base):
    __tablename__ = "players"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    
    team_name: Mapped[str] = mapped_column(String(100), index=True)
    api_id: Mapped[int | None] = mapped_column(Integer, unique=True, nullable=True)
    
    name: Mapped[str] = mapped_column(String(150))
    position: Mapped[str | None] = mapped_column(String(50))
    number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    age: Mapped[int | None] = mapped_column(Integer, nullable=True)
    nacionality: Mapped[str | None] = mapped_column(String(100), nullable=True)
    club: Mapped[str | None] = mapped_column(String(150), nullable=True)
    photo_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())
    
    def __repr__(self):
        return f"<Player {self.name} | Team: {self.team_name} | Position: {self.position}>"