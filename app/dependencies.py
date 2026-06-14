from dataclasses import dataclass

from fastapi import Depends, Request
from sqlalchemy.orm import Session

from app.db.models.team import Team
from app.db.session import get_db
from app.ml.loader import MLArtifacts


def get_live_elo(db: Session = Depends(get_db)) -> dict[str, float]:
    """
    ELO vigente por equipo desde teams.elo_rating, keyed en minúsculas (igual que
    get_team_elo). Refleja los ajustes de la última re-simulación; permite que las
    predicciones de partidos pendientes sean consistentes con la simulación.
    Si aún no hay valores, queda vacío (se usa el ELO base del artifact).
    """
    rows = db.query(Team.name, Team.elo_rating).filter(Team.elo_rating.isnot(None)).all()
    return {name.lower(): float(elo) for name, elo in rows}


def get_artifacts(request: Request) -> MLArtifacts:
    """
    Retorna la instancia de MLArtifacts almacenada en app.state.

    main.py la guarda durante el lifespan:
        app.state.artifacts = load_artifacts()

    De esta forma los modelos se cargan UNA sola vez y se comparten
    entre todos los requests sin reinicialización.
    """
    return request.app.state.artifacts

@dataclass
class Pagination:
    skip: int  = 0
    limit: int = 50


def get_pagination(skip: int = 0, limit: int = 50) -> Pagination:
    """
    Parámetros de paginación extraídos del query string.
    Ejemplo: GET /teams?skip=0&limit=20
    El límite se fuerza a máximo 100 para evitar respuestas demasiado grandes.
    """
    return Pagination(skip=skip, limit=min(limit, 100))


def get_current_user():
    """
    verificará el JWT del header Authorization y retornará
    el usuario autenticado. Por ahora no hace nada.
    """
    pass


__all__ = [
    "get_db",
    "get_artifacts",
    "get_live_elo",
    "get_pagination",
    "get_current_user",
    "Pagination",
]