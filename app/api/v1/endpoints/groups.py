from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.models.match import Match
from app.dependencies import get_db, get_artifacts
from app.ml.loader import MLArtifacts
from app.ml.predictor import predict_match
from app.schemas import PredictionSummary

router = APIRouter()

_LABELS = {"H": "Victoria local", "D": "Empate", "A": "Victoria visitante"}


@router.get("/groups", tags=["Groups"])
def get_groups(
    db: Session = Depends(get_db),
    artifacts: MLArtifacts = Depends(get_artifacts),
):
    """
    Retorna los partidos de fase de grupos organizados por grupo (A-L),
    con las probabilidades predichas para cada partido.
    """
    matches = (
        db.query(Match)
        .filter(Match.stage == "Group Stage", Match.group.isnot(None))
        .order_by(Match.group, Match.date)
        .all()
    )

    groups: dict[str, list] = {}
    for m in matches:
        pred_result = predict_match(m.home_team, m.away_team, artifacts)
        prediction = None
        if pred_result:
            best = max(
                {"H": pred_result.p_home_win, "D": pred_result.p_draw, "A": pred_result.p_away_win},
                key=lambda k: {"H": pred_result.p_home_win, "D": pred_result.p_draw, "A": pred_result.p_away_win}[k],
            )
            prediction = PredictionSummary(
                p_home_win=pred_result.p_home_win,
                p_draw=pred_result.p_draw,
                p_away_win=pred_result.p_away_win,
                prediction=best,
                prediction_label=_LABELS[best],
            )

        groups.setdefault(m.group, []).append({
            "id": m.id,
            "home_team": m.home_team,
            "away_team": m.away_team,
            "date": m.date.isoformat(),
            "city": m.city,
            "prediction": prediction.model_dump() if prediction else None,
        })

    # Retornamos los grupos ordenados alfabéticamente
    return {"groups": dict(sorted(groups.items()))}