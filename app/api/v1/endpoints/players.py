from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.exceptions import PlayersNotFoundException
from app.db.models.player import Player
from app.db.session import get_db
from app.schemas.player import PlayerResponse

router = APIRouter()

@router.get("/players/{team_name}", response_model=list[PlayerResponse], tags=["Players"])
def get_players_by_team(team_name: str, db: Session = Depends(get_db)):
    """Retorna el plantel de un equipo dado su nombre."""
    team_name = team_name.strip().title()
    players = db.query(Player).filter(Player.team_name == team_name).all()
    if not players:
        raise PlayersNotFoundException(f"No se encontró el equipo con nombre {team_name}")
    return players