import time
import logging

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.config import settings
from app.core.exceptions import TeamNotFoundException
from app.db.models.team import Team
from app.dependencies import get_db, get_pagination, Pagination
from app.schemas import TeamResponse

logger = logging.getLogger(__name__)
router = APIRouter()

# Cache en memoria para planteles: {slug: (timestamp, data)}
# Se invalida automáticamente pasadas 24 horas.
_SQUAD_CACHE: dict[str, tuple[float, list]] = {}
_SQUAD_CACHE_TTL = 60 * 60 * 24  # 24 horas en segundos


@router.get("/teams", response_model=list[TeamResponse], tags=["Teams"])
def list_teams(
    db: Session = Depends(get_db),
    pagination: Pagination = Depends(get_pagination),
):
    """Lista todos los equipos ordenados por Elo descendente."""
    return (
        db.query(Team)
        .order_by(Team.elo_rating.desc().nullslast())
        .offset(pagination.skip)
        .limit(pagination.limit)
        .all()
    )


@router.get("/teams/{slug}", response_model=TeamResponse, tags=["Teams"])
def get_team(slug: str, db: Session = Depends(get_db)):
    """Retorna el detalle de un equipo por su slug."""
    team = db.query(Team).filter(Team.slug == slug).first()
    if not team:
        raise TeamNotFoundException(f"No se encontró el equipo '{slug}'")
    return team


@router.get("/teams/{slug}/squad", tags=["Teams"])
def get_squad(slug: str, db: Session = Depends(get_db)):
    """
    Retorna el plantel del equipo consumiendo por un endpoint externo.
    La respuesta se cachea en memoria por 24 horas.
    Requiere la variable de entorno FOOTBALL_DATA_API_KEY.
    """
    # Verificamos que el equipo exista
    team = db.query(Team).filter(Team.slug == slug).first()
    if not team:
        raise TeamNotFoundException(f"No se encontró el equipo '{slug}'")

    api_key = getattr(settings, "football_data_api_key", None)
    if not api_key:
        return {
            "team": team.name,
            "squad": [],
            "note": "FOOTBALL_DATA_API_KEY no configurada.",
        }

    # Cache hit
    cached = _SQUAD_CACHE.get(slug)
    if cached and (time.time() - cached[0]) < _SQUAD_CACHE_TTL:
        return {"team": team.name, "squad": cached[1], "cached": True}

    # TODO: implementar _get_fd_team_id(slug) con el mapa de IDs.
    return {
        "team": team.name,
        "squad": [],
        "note": "Integración con football-data.org pendiente de configurar.",
    }