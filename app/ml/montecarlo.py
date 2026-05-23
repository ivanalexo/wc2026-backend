# =============================================================================
# app/ml/montecarlo.py — Resultados de simulación Monte Carlo
# =============================================================================
# Los resultados ya fueron calculados por 05_montecarlo.py (10,000 iteraciones).
# Este módulo solo los sirve — no re-simula en cada request.
# =============================================================================

from __future__ import annotations

import pandas as pd

from app.ml.loader import MLArtifacts


def get_all_probabilities(artifacts: MLArtifacts) -> list[dict]:
    """Retorna las probabilidades de todos los equipos ordenadas por P(Campeón)."""
    df = artifacts.montecarlo.copy()

    # Normalizamos los nombres de columna a minúsculas sin espacios para
    # no depender del nombre exacto que usó 05_montecarlo.py.
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    # Buscamos la columna de campeón de forma flexible
    champ_col = _find_col(df, ["p_champion", "p_campeon", "champion", "campeon"])
    if champ_col:
        df = df.sort_values(champ_col, ascending=False)

    return df.to_dict(orient="records")


def get_team_probabilities(team: str, artifacts: MLArtifacts) -> dict | None:
    """Retorna las probabilidades de un equipo específico."""
    df = artifacts.montecarlo.copy()
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    team_col = _find_col(df, ["team", "equipo"])
    if team_col is None:
        return None

    row = df[df[team_col].str.lower() == team.lower()]
    if row.empty:
        return None

    return row.iloc[0].to_dict()


def get_top_n(n: int, artifacts: MLArtifacts) -> list[dict]:
    """Retorna los N equipos con mayor probabilidad de ser campeones."""
    all_probs = get_all_probabilities(artifacts)
    return all_probs[:n]


# =============================================================================
# Helper privado
# =============================================================================

def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Busca la primera columna del DataFrame que coincida con algún candidato."""
    for c in candidates:
        if c in df.columns:
            return c
    return None