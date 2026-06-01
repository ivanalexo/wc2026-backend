import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import router as api_router
from app.config import settings
from app.core.exceptions import (
    AppException,
    app_exception_handler,
    unhandled_exception_handler,
)
from app.ml.loader import load_artifacts

logging.basicConfig(
    level=logging.DEBUG if settings.app_debug else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info("Cargando artefactos ML...")
    try:
        app.state.artifacts = load_artifacts()
        logger.info("Artefactos ML cargados.")
    except FileNotFoundError as e:
        logger.warning("Artefactos no encontrados — modo degradado: %s", e)
        app.state.artifacts = None

    app.state.squad_cache = {}

    yield
    logger.info("Apagando la aplicación.")


app = FastAPI(
    title="WC 2026 Predictor API",
    description="Predicción del Mundial FIFA 2026 con XGBoost y Monte Carlo.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_exception_handler(AppException, app_exception_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)

app.include_router(api_router, prefix="/api/v1")