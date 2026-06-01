from fastapi import APIRouter, Depends

from app.core.exceptions import PredictionUnavailableException
from app.dependencies import get_artifacts
from app.ml.loader import MLArtifacts
from app.ml.predictor import resolve_team_name

router = APIRouter()


def _compute_h2h(
    team1: str,
    team2: str,
    artifacts: MLArtifacts,
) -> dict:
    """
    Calcula el H2H entre dos equipos desde el historial completo.
    Los resultados son conteos reales (no ponderados por torneo).
    Retorna None si no hay datos disponibles.
    """
    hist = artifacts.historical_results
    if hist is None:
        return None

    # Filtrar todos los partidos entre los dos equipos
    mask = (
        ((hist["home_team"] == team1) & (hist["away_team"] == team2)) |
        ((hist["home_team"] == team2) & (hist["away_team"] == team1))
    )
    matches = hist[mask].sort_values("date", ascending=False).copy()

    if matches.empty:
        return None

    team1_wins = 0
    draws      = 0
    team2_wins = 0

    for _, row in matches.iterrows():
        result = row.get("result")
        if result is None:
            continue

        if row["home_team"] == team1:
            # team1 es local
            if result == "H":
                team1_wins += 1
            elif result == "D":
                draws += 1
            else:
                team2_wins += 1
        else:
            if result == "A":
                team1_wins += 1
            elif result == "D":
                draws += 1
            else:
                team2_wins += 1

    total = team1_wins + draws + team2_wins

    # Últimos 5 partidos para mostrar en el frontend
    recent = []
    for _, row in matches.head(5).iterrows():
        home_score = row.get("home_score")
        away_score = row.get("away_score")
        recent.append({
            "date":       str(row["date"].date()) if hasattr(row["date"], "date") else str(row["date"]),
            "home_team":  row["home_team"],
            "away_team":  row["away_team"],
            "home_score": int(home_score) if home_score is not None and home_score == home_score else None,
            "away_score": int(away_score) if away_score is not None and away_score == away_score else None,
            "tournament": row.get("tournament", ""),
        })

    return {
        "team1":           team1,
        "team2":           team2,
        "total_matches":   total,
        "team1_wins":      team1_wins,
        "draws":           draws,
        "team2_wins":      team2_wins,
        "team1_win_rate":  round(team1_wins / total, 3) if total > 0 else 0.5,
        "recent_matches":  recent,
        "note": "Conteos históricos reales sin ponderación por torneo.",
    }


@router.get("/stats/h2h/{team1}/{team2}", tags=["Stats"])
def head_to_head(
    team1: str,
    team2: str,
    artifacts: MLArtifacts = Depends(get_artifacts),
):
    """
    Retorna el historial H2H entre dos equipos.
    Incluye conteos de victorias, empates y los 5 partidos más recientes.
    """
    if artifacts.historical_results is None:
        raise PredictionUnavailableException(
            "El historial de partidos no está disponible. "
            "Copia clean_results_historical.csv a la carpeta artifacts/."
        )

    t1 = resolve_team_name(team1, artifacts)
    t2 = resolve_team_name(team2, artifacts)

    if t1.lower() == t2.lower():
        raise PredictionUnavailableException("Los dos equipos deben ser diferentes.")

    result = _compute_h2h(t1, t2, artifacts)

    if result is None:
        raise PredictionUnavailableException(
            f"No hay partidos históricos registrados entre '{t1}' y '{t2}'."
        )

    return result