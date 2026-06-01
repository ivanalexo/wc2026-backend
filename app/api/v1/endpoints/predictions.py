import math

import numpy as np
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.exceptions import PredictionUnavailableException
from app.db.models.predictions_cache import PredictionsCache
from app.dependencies import get_db, get_artifacts
from app.ml.explainer import explain_prediction
from app.ml.loader import MLArtifacts
from app.ml.predictor import predict_match, predict_score, resolve_team_name, get_team_elo
from app.schemas import (
    PredictionRequest,
    PredictionResponse,
    ShapExplanation,
    ScorePredictionResponse,
    ScoreEntryResponse,
)

router = APIRouter()


def _sanitize_for_json(d: dict) -> dict:
    """
    Convierte tipos numpy y NaN/Inf a tipos Python serializables por JSON.
    """
    result = {}
    for k, v in d.items():
        if isinstance(v, (np.integer,)):
            result[k] = int(v)
        elif isinstance(v, (np.floating,)):
            result[k] = None if (np.isnan(v) or np.isinf(v)) else float(v)
        elif isinstance(v, np.bool_):
            result[k] = bool(v)
        elif isinstance(v, float):
            result[k] = None if (math.isnan(v) or math.isinf(v)) else v
        else:
            result[k] = v
    return result


@router.post("/predict/match", response_model=PredictionResponse, tags=["Predictions"])
def predict_match_endpoint(
    body: PredictionRequest,
    db: Session = Depends(get_db),
    artifacts: MLArtifacts = Depends(get_artifacts),
):
    """
    Predice el resultado de un partido (H/D/A) usando el modelo XGBoost.
    Los nombres de equipo son case-insensitive: "jordan" = "Jordan" = "JORDAN".
    La predicción se cachea — el mismo par no se recalcula dos veces.
    """
    # Normalizamos nombres antes de consultar la caché
    home = resolve_team_name(body.home_team, artifacts)
    away = resolve_team_name(body.away_team, artifacts)

    # --- Cache hit ----------------------------------------------------------
    cached = (
        db.query(PredictionsCache)
        .filter(
            PredictionsCache.home_team == home,
            PredictionsCache.away_team == away,
        )
        .first()
    )
    if cached:
        explanation = None
        if cached.shap_explanation:
            explanation = ShapExplanation(**cached.shap_explanation)
        return PredictionResponse(
            home_team=cached.home_team,
            away_team=cached.away_team,
            p_home_win=cached.p_home_win,
            p_draw=cached.p_draw,
            p_away_win=cached.p_away_win,
            prediction=cached.prediction,
            home_elo=get_team_elo(home, artifacts),
            away_elo=get_team_elo(away, artifacts),
            elo_diff=None,
            explanation=explanation,
            cached=True,
        )

    # --- Predicción ---------------------------------------------------------
    result = predict_match(home, away, artifacts)
    if result is None:
        raise PredictionUnavailableException(
            f"No hay datos de Elo para '{home}' o '{away}'"
        )

    # --- Explicación SHAP ---------------------------------------------------
    explanation = explain_prediction(
        features=result.features,
        prediction=result.prediction,
        home_team=home,
        away_team=away,
        artifacts=artifacts,
    )

    # --- Guardar en cache ---------------------------------------------------
    db.merge(
        PredictionsCache(
            home_team=home,
            away_team=away,
            p_home_win=result.p_home_win,
            p_draw=result.p_draw,
            p_away_win=result.p_away_win,
            prediction=result.prediction,
            features=_sanitize_for_json(result.features),
            shap_explanation=explanation.model_dump(),
        )
    )
    db.commit()

    return PredictionResponse(
        home_team=result.home_team,
        away_team=result.away_team,
        p_home_win=result.p_home_win,
        p_draw=result.p_draw,
        p_away_win=result.p_away_win,
        prediction=result.prediction,
        home_elo=result.home_elo,
        away_elo=result.away_elo,
        elo_diff=result.elo_diff,
        explanation=explanation,
        cached=False,
    )


@router.post("/predict/score", response_model=ScorePredictionResponse, tags=["Predictions"])
def predict_score_endpoint(
    body: PredictionRequest,
    artifacts: MLArtifacts = Depends(get_artifacts),
):
    """
    Predice el marcador más probable usando Poisson + Monte Carlo.
    Los nombres de equipo son case-insensitive.
    """
    result = predict_score(body.home_team, body.away_team, artifacts)
    if result is None:
        raise PredictionUnavailableException(
            f"No hay datos suficientes para '{body.home_team}' vs '{body.away_team}'"
        )

    return ScorePredictionResponse(
        home_team=result.home_team,
        away_team=result.away_team,
        predicted_home_goals=result.predicted_home_goals,
        predicted_away_goals=result.predicted_away_goals,
        expected_home_goals=result.expected_home_goals,
        expected_away_goals=result.expected_away_goals,
        p_home_win=result.p_home_win,
        p_draw=result.p_draw,
        p_away_win=result.p_away_win,
        top_scores=[
            ScoreEntryResponse(score=e.score, probability=e.probability)
            for e in result.top_scores
        ],
        n_simulations=result.n_simulations,
    )