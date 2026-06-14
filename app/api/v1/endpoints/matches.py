from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.exceptions import MatchNotFoundException
from app.db.models.match import Match
from app.dependencies import get_db, get_artifacts, get_live_elo, get_pagination, Pagination
from app.ml.loader import MLArtifacts
from app.ml.predictor import predict_match
from app.schemas import MatchWithPrediction, PredictionSummary

router = APIRouter()


def _build_prediction_summary(
    home_team: str,
    away_team: str,
    artifacts: MLArtifacts,
    elo_override: dict[str, float] | None = None,
) -> PredictionSummary | None:
    """
    Genera el resumen de probabilidades para embeber en cada partido.

    `elo_override` solo debe pasarse para partidos NO finalizados: los pendientes
    reflejan el ELO vigente, mientras que los jugados se quedan con el ELO base
    (la predicción original, evitando que el resultado contamine su propia
    predicción vía el ELO que él mismo modificó).
    """
    result = predict_match(home_team, away_team, artifacts, elo_override=elo_override)
    if result is None:
        return None
    best = max(
        {"H": result.p_home_win, "D": result.p_draw, "A": result.p_away_win},
        key=lambda k: {"H": result.p_home_win, "D": result.p_draw, "A": result.p_away_win}[k],
    )
    labels = {"H": "Victoria local", "D": "Empate", "A": "Victoria visitante"}
    return PredictionSummary(
        p_home_win=result.p_home_win,
        p_draw=result.p_draw,
        p_away_win=result.p_away_win,
        prediction=best,
        prediction_label=labels[best],
    )


@router.get("/fixtures", response_model=list[MatchWithPrediction], tags=["Fixtures"])
def list_fixtures(
    group: str | None = None,
    stage: str | None = None,
    upcoming: bool = False,
    db: Session = Depends(get_db),
    artifacts: MLArtifacts = Depends(get_artifacts),
    live_elo: dict[str, float] = Depends(get_live_elo),
    pagination: Pagination = Depends(get_pagination),
):
    """
    Lista los 72 partidos del Mundial 2026 con probabilidades embebidas.
    Filtros opcionales: group (A-L), stage (ej: 'Group Stage') y
    upcoming (excluye partidos ya finalizados, mostrando solo los próximos).
    """
    query = db.query(Match).order_by(Match.date)

    if group:
        query = query.filter(Match.group == group.upper())
    if stage:
        query = query.filter(Match.stage == stage)
    if upcoming:
        query = query.filter(Match.status != "finished")

    matches = query.offset(pagination.skip).limit(pagination.limit).all()

    return [
        MatchWithPrediction(
            **{c.name: getattr(m, c.name) for c in Match.__table__.columns},
            prediction=_build_prediction_summary(
                m.home_team, m.away_team, artifacts,
                elo_override=None if m.status == "finished" else live_elo,
            ),
        )
        for m in matches
    ]


@router.get("/fixtures/{match_id}", response_model=MatchWithPrediction, tags=["Fixtures"])
def get_fixture(
    match_id: int,
    db: Session = Depends(get_db),
    artifacts: MLArtifacts = Depends(get_artifacts),
    live_elo: dict[str, float] = Depends(get_live_elo),
):
    """Retorna el detalle de un partido con su predicción."""
    match = db.query(Match).filter(Match.id == match_id).first()
    if not match:
        raise MatchNotFoundException(f"Partido con id {match_id} no encontrado")

    return MatchWithPrediction(
        **{c.name: getattr(match, c.name) for c in Match.__table__.columns},
        prediction=_build_prediction_summary(
            match.home_team, match.away_team, artifacts,
            elo_override=None if match.status == "finished" else live_elo,
        ),
    )