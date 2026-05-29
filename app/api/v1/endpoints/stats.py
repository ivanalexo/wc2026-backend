from fastapi import APIRouter, Depends

from app.core.exceptions import PredictionUnavailableException
from app.dependencies import get_artifacts
from app.ml.loader import MLArtifacts

router = APIRouter()


@router.get("/stats/h2h/{team1}/{team2}", tags=["Stats"])
def head_to_head(
    team1: str,
    team2: str,
    artifacts: MLArtifacts = Depends(get_artifacts),
):
    """
    Retorna el historial H2H ponderado entre dos equipos extraído de los
    features pre-computados del fixture 2026.
    Los valores H2H fueron calculados en 02_features.py sobre todos los
    partidos históricos previos al Mundial 2026.
    """
    fixture = artifacts.fixture_features

    # Buscamos el par en cualquiera de los dos órdenes
    mask_direct  = (fixture["home_team"] == team1) & (fixture["away_team"] == team2)
    mask_inverse = (fixture["home_team"] == team2) & (fixture["away_team"] == team1)

    row_direct  = fixture[mask_direct].head(1)
    row_inverse = fixture[mask_inverse].head(1)

    if row_direct.empty and row_inverse.empty:
        raise PredictionUnavailableException(
            f"No hay datos H2H para '{team1}' vs '{team2}'. "
            "Asegúrate de usar los nombres exactos del fixture."
        )

    # Tomamos la primera fila que encontremos y normalizamos la perspectiva
    if not row_direct.empty:
        row       = row_direct.iloc[0]
        home_wins = float(row.get("h2h_home_wins", 0))
        away_wins = float(row.get("h2h_away_wins", 0))
    else:
        row       = row_inverse.iloc[0]
        # Al invertir, los roles home/away se intercambian
        home_wins = float(row.get("h2h_away_wins", 0))
        away_wins = float(row.get("h2h_home_wins", 0))

    draws = float(row.get("h2h_draws", 0))
    total = float(row.get("h2h_total", 0))

    return {
        "team1": team1,
        "team2": team2,
        "total_matches_weighted": round(total, 2),
        "team1_wins_weighted":    round(home_wins, 2),
        "draws_weighted":         round(draws, 2),
        "team2_wins_weighted":    round(away_wins, 2),
        "team1_win_rate":         round(home_wins / total, 3) if total > 0 else 0.5,
        "note": (
            "Los valores son ponderados: partidos de Mayor importancia "
            "(WC=1.0) pesan más que amistosos (0.25)."
        ),
    }