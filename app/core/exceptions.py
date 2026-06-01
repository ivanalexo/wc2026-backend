import logging

from fastapi import Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

class AppException(Exception):
    """Base para todas las excepciones de la aplicación."""
    status_code: int = 500
    detail: str = "Error interno del servidor"

    def __init__(self, detail: str | None = None):
        self.detail = detail or self.__class__.detail
        super().__init__(self.detail)


class TeamNotFoundException(AppException):
    status_code = 404
    detail = "Equipo no encontrado"


class MatchNotFoundException(AppException):
    status_code = 404
    detail = "Partido no encontrado"


class PredictionUnavailableException(AppException):
    """Se lanza cuando no hay suficiente información para predecir.
    Por ejemplo: ninguno de los dos equipos tiene dato de Elo."""
    status_code = 422
    detail = "No hay suficiente información para generar una predicción"


class ArtifactsNotLoadedException(AppException):
    """Se lanza si get_artifacts() es llamado antes de que main.py
    haya ejecutado load_artifacts() en el lifespan."""
    status_code = 503
    detail = "Los modelos ML no están disponibles. Intenta más tarde."

def _error_body(status_code: int, detail: str) -> dict:
    return {"error": {"status": status_code, "detail": detail}}


async def app_exception_handler(
    request: Request, exc: AppException
) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content=_error_body(exc.status_code, exc.detail),
    )


async def unhandled_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Captura cualquier excepción no manejada para evitar que FastAPI
    exponga stack traces en producción."""
    logger.exception("Excepción no manejada en %s %s", request.method, request.url)
    return JSONResponse(
        status_code=500,
        content=_error_body(500, "Error interno del servidor"),
    )