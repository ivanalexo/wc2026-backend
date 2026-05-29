from fastapi import APIRouter, Depends

from app.dependencies import get_artifacts
from app.ml import montecarlo
from app.ml.loader import MLArtifacts

router = APIRouter()


@router.get("/simulate/tournament", tags=["Simulation"])
def get_tournament_simulation(
    top: int = 10,
    artifacts: MLArtifacts = Depends(get_artifacts),
):
    """
    Retorna los resultados de la simulación Monte Carlo del torneo completo
    (10,000 iteraciones pre-calculadas en 05_montecarlo.py).

    No re-simula en cada request — sirve los resultados ya calculados.
    Parámetro `top`: cuántos equipos retornar ordenados por P(Campeón).
    """
    top = min(top, 48)
    results = montecarlo.get_top_n(top, artifacts)
    return {
        "simulation": {
            "n_iterations": 10_000,
            "source": "pre-computed (05_montecarlo.py)",
        },
        "top_teams": results,
    }


@router.get("/simulate/tournament/{team}", tags=["Simulation"])
def get_team_simulation(
    team: str,
    artifacts: MLArtifacts = Depends(get_artifacts),
):
    """Retorna las probabilidades por ronda para un equipo específico."""
    result = montecarlo.get_team_probabilities(team, artifacts)
    if result is None:
        return {"team": team, "probabilities": None, "note": "Equipo no encontrado en la simulación."}
    return {"team": team, "probabilities": result}