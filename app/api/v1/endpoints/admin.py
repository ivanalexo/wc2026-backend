import logging

from fastapi import APIRouter, Depends, Header, HTTPException, status
from sqlalchemy.orm import Session

from app.config import settings
from app.dependencies import get_db
from app.services.results_sync import SyncResult, sync_wc_results

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin", tags=["Admin"])


def _verify_secret(x_sync_secret: str | None = Header(default=None)) -> None:
    """Valida la cabecera X-Sync-Secret contra SYNC_SECRET del entorno."""
    configured = settings.sync_secret
    if not configured:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="SYNC_SECRET no configurado en el servidor",
        )
    if x_sync_secret != configured:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="X-Sync-Secret inválido",
        )


@router.post(
    "/sync",
    response_model=dict,
    include_in_schema=False,  # No aparece en Swagger/OpenAPI — solo accesible si conoces la ruta y el secret
)
def trigger_sync(
    db: Session = Depends(get_db),
    _: None = Depends(_verify_secret),
) -> dict:
    logger.info("Sync de resultados iniciado via endpoint /admin/sync")
    result: SyncResult = sync_wc_results(db)

    if result.error:
        logger.error("Sync fallido: %s", result.error)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=result.error,
        )

    logger.info(
        "Sync completado — actualizados=%d skipped=%d not_found=%d",
        result.updated, result.skipped, result.not_found,
    )
    return {
        "updated":            result.updated,
        "skipped":            result.skipped,
        "not_found":          result.not_found,
        "requests_available": result.requests_available,
    }
