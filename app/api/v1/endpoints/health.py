from fastapi import APIRouter, Request
from sqlalchemy import text

from app.db.session import SessionLocal

router = APIRouter()


@router.get("/health", tags=["Health"])
def health_check(request: Request) -> dict:
    """
    Verifica el estado de los tres componentes críticos:
    la API, la base de datos y los artefactos ML.
    """
    # --- Base de datos ------------------------------------------------------
    db_status = "ok"
    try:
        with SessionLocal() as db:
            db.execute(text("SELECT 1"))
    except Exception:
        db_status = "unavailable"

    # --- Artefactos ML ------------------------------------------------------
    artifacts = getattr(request.app.state, "artifacts", None)
    ml_status = "ok" if artifacts is not None else "unavailable"

    overall = "ok" if db_status == "ok" and ml_status == "ok" else "degraded"

    return {
        "status": overall,
        "components": {
            "api": "ok",
            "database": db_status,
            "ml_artifacts": ml_status,
        },
    }