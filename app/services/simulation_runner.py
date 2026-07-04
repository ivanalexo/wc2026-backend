from __future__ import annotations

import logging

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models.match import Match
from app.db.models.predictions_cache import PredictionsCache
from app.db.models.simulation_result import SimulationResult
from app.db.models.team import Team
from app.db.session import SessionLocal
from app.ml.elo import FinishedMatch, recompute_elo
from app.ml.loader import MLArtifacts
from app.ml.simulation import (
    N_SIMULATIONS_DEFAULT,
    base_team_elo,
    build_context,
    run_simulation,
)

logger = logging.getLogger(__name__)


def _finished_matches(db: Session) -> list[Match]:
    """Partidos finalizados con marcador, en orden cronologico (para el replay ELO)."""
    return list(
        db.scalars(
            select(Match)
            .where(
                Match.status == "finished",
                Match.home_score.isnot(None),
                Match.away_score.isnot(None),
            )
            .order_by(Match.date.asc())
        )
    )


def build_played_results(matches: list[Match]) -> dict[frozenset, dict]:
    played: dict[frozenset, dict] = {}
    for m in matches:
        if m.stage != "Group Stage":
            continue
        key = frozenset((m.home_team, m.away_team))
        if m.home_team == m.away_team or len(key) != 2:
            continue
        played[key] = {
            "home": m.home_team,
            "away": m.away_team,
            "home_score": int(m.home_score),
            "away_score": int(m.away_score),
        }
    return played


def regenerate_simulation(
    db: Session,
    artifacts: MLArtifacts,
    n_simulations: int = N_SIMULATIONS_DEFAULT,
) -> dict:
    """
    Recalcula ELO + simulación condicionada y persiste todo en la DB.
    Retorna un resumen con el top-10 y el conteo de partidos jugados.
    """
    finished = _finished_matches(db)

    base_elo = base_team_elo(artifacts)
    replay = [
        FinishedMatch(m.home_team, m.away_team, int(m.home_score), int(m.away_score))
        for m in finished
    ]
    current_elo = recompute_elo(base_elo, replay)

    played = build_played_results(finished)
    logger.info("Re-simulando: %d jugados, n=%d", len(played), n_simulations)
    ctx = build_context(artifacts, elo_override=current_elo)
    df = run_simulation(artifacts, played_results=played, n_simulations=n_simulations, ctx=ctx)

    db.query(SimulationResult).delete()
    db.bulk_save_objects([
        SimulationResult(
            team=row["team"],
            elo=float(row["elo"]),
            p_qualify=float(row["p_qualify"]),
            p_reach_r16=float(row["p_reach_r16"]),
            p_reach_qf=float(row["p_reach_qf"]),
            p_reach_sf=float(row["p_reach_sf"]),
            p_reach_final=float(row["p_reach_final"]),
            p_champion=float(row["p_champion"]),
        )
        for _, row in df.iterrows()
    ])

    for team in db.scalars(select(Team)):
        if team.name in current_elo:
            team.elo_rating = round(float(current_elo[team.name]), 1)

    db.query(PredictionsCache).delete()

    db.commit()
    logger.info("Simulación persistida y caché de predicciones vaciada")

    top = df.head(10)[["team", "p_qualify", "p_champion"]].to_dict(orient="records")
    return {"played_matches": len(played), "n_simulations": n_simulations, "top_10": top}


def regenerate_simulation_task(artifacts: MLArtifacts) -> None:
    """Wrapper para BackgroundTasks: gestiona su propia sesión de DB."""
    db = SessionLocal()
    try:
        regenerate_simulation(db, artifacts)
    except Exception:
        logger.exception("Fallo en la re-simulación de fondo")
    finally:
        db.close()
