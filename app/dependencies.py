# =============================================================================
# app/dependencies.py — Dependencias compartidas entre endpoints
# =============================================================================
# Uso en cualquier endpoint:
#
#   from app.dependencies import get_db, get_artifacts, Pagination
#
#   @router.get("/fixtures")
#   def list_fixtures(
#       db: Session = Depends(get_db),
#       artifacts: MLArtifacts = Depends(get_artifacts),
#       pagination: Pagination = Depends(get_pagination),
#   ): ...
#
# FastAPI resuelve cada Depends() una vez por request y gestiona el ciclo
# de vida (por ejemplo, cierra la sesión de DB al finalizar el request).
# =============================================================================

from dataclasses import dataclass

from fastapi import Depends, Request
from sqlalchemy.orm import Session

from app.db.session import get_db          # re-exportamos para import único
from app.ml.loader import MLArtifacts


# =============================================================================
# Dependencia ML — artefactos cargados al startup
# =============================================================================

def get_artifacts(request: Request) -> MLArtifacts:
    """
    Retorna la instancia de MLArtifacts almacenada en app.state.

    main.py la guarda durante el lifespan:
        app.state.artifacts = load_artifacts()

    De esta forma los modelos se cargan UNA sola vez y se comparten
    entre todos los requests sin reinicialización.
    """
    return request.app.state.artifacts


# =============================================================================
# Dependencia de paginación — para endpoints que listan colecciones
# =============================================================================

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


# =============================================================================
# Stub de autenticación — se implementa en Fase 4
# =============================================================================
# Se define ahora para que los endpoints que lo necesitarán en Fase 4
# puedan importarlo desde ya, evitando cambiar sus firmas después.

def get_current_user():
    """
    Fase 4: verificará el JWT del header Authorization y retornará
    el usuario autenticado. Por ahora no hace nada.
    """
    pass


__all__ = [
    "get_db",
    "get_artifacts",
    "get_pagination",
    "get_current_user",
    "Pagination",
]