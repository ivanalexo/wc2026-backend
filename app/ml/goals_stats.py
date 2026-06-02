from __future__ import annotations

import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Tipos
# ─────────────────────────────────────────────────────────────────────────────

class TeamGoalStats:
    """Estadísticas de goles por equipo (promedio por partido)."""
    __slots__ = ("avg_gf", "avg_ga", "n_games")

    def __init__(self, avg_gf: float, avg_ga: float, n_games: int) -> None:
        self.avg_gf  = avg_gf    # goles a favor promedio
        self.avg_ga  = avg_ga    # goles en contra promedio
        self.n_games = n_games   # partidos en el dataset

    def __repr__(self) -> str:
        return (
            f"TeamGoalStats(avg_gf={self.avg_gf:.2f}, "
            f"avg_ga={self.avg_ga:.2f}, n_games={self.n_games})"
        )

def build_team_goals_lookup(
    goalscorers: pd.DataFrame,
    matches: pd.DataFrame,
    *,
    home_team_col: str  = "home_team",
    away_team_col: str  = "away_team",
    home_score_col: str = "home_score",
    away_score_col: str = "away_score",
    own_goal_col: str | None = "own_goal",
    min_games: int = 1,
) -> dict[str, TeamGoalStats]:
    """
    Construye un lookup {equipo_lowercase → TeamGoalStats} a partir
    del dataset de goalscorers y del dataset de resultados de partidos.

    Parámetros
    ----------
    goalscorers : DataFrame
        Una fila por gol. Columnas esperadas:
            home_team, away_team[, own_goal]
        Si ``own_goal_col`` existe y es True, el gol se asigna al equipo
        que no anotó (error en propia meta).

    matches : DataFrame
        Una fila por partido. Columnas esperadas:
            home_team, away_team, home_score, away_score

    own_goal_col : str | None
        Columna booleana para autogoles. Pass None si el dataset no la tiene.

    min_games : int
        Equipos con menos partidos que este umbral son excluidos del lookup.

    Devuelve
    --------
    dict[str, TeamGoalStats]
        Clave: nombre del equipo en minúsculas (para lookup case-insensitive).
    """
    m = matches[[home_team_col, away_team_col, home_score_col, away_score_col]].dropna().copy()
    m[home_team_col]  = m[home_team_col].str.strip().str.lower()
    m[away_team_col]  = m[away_team_col].str.strip().str.lower()
    m[home_score_col] = pd.to_numeric(m[home_score_col], errors="coerce")
    m[away_score_col] = pd.to_numeric(m[away_score_col], errors="coerce")
    m = m.dropna(subset=[home_score_col, away_score_col])

    # Conteo de goles por equipo desde goalscorers
    gf_overrides = _count_goals_from_goalscorers(
        goalscorers,
        home_team_col=home_team_col,
        away_team_col=away_team_col,
        own_goal_col=own_goal_col,
    )

    all_teams = pd.concat([m[home_team_col], m[away_team_col]]).unique()
    lookup: dict[str, TeamGoalStats] = {}

    for team in all_teams:
        home_rows = m[m[home_team_col] == team]
        away_rows = m[m[away_team_col] == team]

        goals_scored   = pd.concat([home_rows[home_score_col], away_rows[away_score_col]])
        goals_conceded = pd.concat([home_rows[away_score_col], away_rows[home_score_col]])

        n_games = len(goals_scored)
        if n_games < min_games:
            continue

        total_scored = gf_overrides.get(team, None)
        avg_gf = (total_scored / n_games) if total_scored is not None else float(goals_scored.mean())
        avg_ga = float(goals_conceded.mean())

        lookup[team] = TeamGoalStats(
            avg_gf=round(avg_gf, 4),
            avg_ga=round(avg_ga, 4),
            n_games=n_games,
        )

    return lookup

def _count_goals_from_goalscorers(
    goalscorers: pd.DataFrame,
    *,
    home_team_col: str,
    away_team_col: str,
    own_goal_col: str | None,
) -> dict[str, int]:
    """
    Cuenta goles anotados por cada equipo (excluyendo autogoles).
    Devuelve {equipo_lower: total_goles}.
    """
    gs = goalscorers.copy()
    gs[home_team_col] = gs[home_team_col].str.strip().str.lower()
    gs[away_team_col] = gs[away_team_col].str.strip().str.lower()

    is_own_goal: pd.Series = pd.Series(False, index=gs.index)
    if own_goal_col and own_goal_col in gs.columns:
        is_own_goal = gs[own_goal_col].fillna(False).astype(bool)

    # Caso A: columna 'team' indica el equipo que anotó
    if "team" in gs.columns:
        gs["team"] = gs["team"].str.strip().str.lower()
        valid = gs[~is_own_goal]
        return valid.groupby("team").size().to_dict()

    return {}

def top_scoring_teams(
    lookup: dict[str, TeamGoalStats],
    n: int = 10,
) -> list[tuple[str, float]]:
    """Devuelve los N equipos con mayor avg_gf, ordenados desc."""
    return sorted(
        ((team, stats.avg_gf) for team, stats in lookup.items()),
        key=lambda x: x[1],
        reverse=True,
    )[:n]