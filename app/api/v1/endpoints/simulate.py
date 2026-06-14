from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models.simulation_result import SimulationResult
from app.dependencies import get_db

router = APIRouter()


def _row_to_dict(r: SimulationResult) -> dict:
    return {
        "team": r.team,
        "elo": r.elo,
        "p_qualify": r.p_qualify,
        "p_reach_r16": r.p_reach_r16,
        "p_reach_qf": r.p_reach_qf,
        "p_reach_sf": r.p_reach_sf,
        "p_reach_final": r.p_reach_final,
        "p_champion": r.p_champion,
    }


@router.get("/simulate/tournament", tags=["Simulation"])
def get_tournament_simulation(
    top: int = 10,
    db: Session = Depends(get_db),
):
    """
    Resultados de la última simulación Monte Carlo (persistida en DB).

    Se regenera de forma event-driven cuando /admin/sync detecta resultados
    nuevos: re-aplica ELO + condiciona la simulación a los partidos jugados.
    Parámetro `top`: cuántos equipos retornar ordenados por P(Campeón).
    """
    top = min(top, 48)
    rows = db.scalars(
        select(SimulationResult)
        .order_by(SimulationResult.p_champion.desc())
        .limit(top)
    ).all()

    updated_at = max((r.updated_at for r in rows), default=None) if rows else None

    return {
        "simulation": {
            "n_teams": len(rows),
            "last_updated": updated_at.isoformat() if updated_at else None,
            "source": "db (regenerada en cada resultado nuevo)",
        },
        "top_teams": [_row_to_dict(r) for r in rows],
    }


@router.get("/simulate/tournament/{team}", tags=["Simulation"])
def get_team_simulation(
    team: str,
    db: Session = Depends(get_db),
):
    """Probabilidades por ronda para un equipo específico."""
    row = db.scalar(
        select(SimulationResult).where(SimulationResult.team.ilike(team))
    )
    if row is None:
        return {"team": team, "probabilities": None, "note": "Equipo no encontrado en la simulación."}
    return {"team": team, "probabilities": _row_to_dict(row)}
