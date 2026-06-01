from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.ml.loader import MLArtifacts, MODEL_FEATURES, CLASS_ORDER

_FALLBACK_ELO: float = 1500.0
_WC_GOALS_PER_TEAM: float = 1.41
_ELO_ALPHA: float = 0.4
N_SIMULATIONS: int = 10_000
TOP_SCORES: int = 5


# =============================================================================
# Dataclasses de retorno
# =============================================================================

@dataclass
class MatchPrediction:
    home_team: str
    away_team: str
    p_home_win: float
    p_draw: float
    p_away_win: float
    prediction: str
    home_elo: float | None
    away_elo: float | None
    elo_diff: float | None
    features: dict
    was_inverted: bool


@dataclass
class ScoreEntry:
    score: str
    probability: float


@dataclass
class ScorePrediction:
    home_team: str
    away_team: str
    predicted_home_goals: int
    predicted_away_goals: int
    expected_home_goals: float
    expected_away_goals: float
    p_home_win: float
    p_draw: float
    p_away_win: float
    top_scores: list[ScoreEntry]
    n_simulations: int = N_SIMULATIONS


# =============================================================================
# Helpers de resolución de nombres
# =============================================================================

def resolve_team_name(name: str, artifacts: MLArtifacts) -> str:
    """
    Convierte cualquier variante de capitalización al nombre canónico del fixture.
    "jordan" → "Jordan"  |  "ARGENTINA" → "Argentina"  |  "saudi arabia" → "Saudi Arabia"
    Si el nombre no existe en el mapa, lo devuelve sin cambios.
    """
    return artifacts.team_name_map.get(name.strip().lower(), name.strip())


def get_team_elo(name: str, artifacts: MLArtifacts) -> float | None:
    """Busca el Elo de un equipo de forma case-insensitive."""
    return artifacts.team_elo.get(name.strip().lower())


# =============================================================================
# predict_match — XGBoost
# =============================================================================

def predict_match(
    home_team: str,
    away_team: str,
    artifacts: MLArtifacts,
    is_neutral: bool = True,
) -> MatchPrediction | None:
    # Normalizamos nombres antes de cualquier operación
    home = resolve_team_name(home_team, artifacts)
    away = resolve_team_name(away_team, artifacts)

    feature_row, was_inverted = _get_features(home, away, artifacts, is_neutral)
    if feature_row is None:
        return None

    X = pd.DataFrame([feature_row])[MODEL_FEATURES]
    X_imputed = artifacts.imputer.transform(X)

    proba     = artifacts.model.predict_proba(X_imputed)[0]
    proba_map = dict(zip(CLASS_ORDER, proba))

    p_home_win = float(proba_map["H"])
    p_draw     = float(proba_map["D"])
    p_away_win = float(proba_map["A"])

    if was_inverted:
        p_home_win, p_away_win = p_away_win, p_home_win

    prediction = max(
        {"H": p_home_win, "D": p_draw, "A": p_away_win},
        key=lambda k: {"H": p_home_win, "D": p_draw, "A": p_away_win}[k],
    )

    home_elo = get_team_elo(home, artifacts)
    away_elo = get_team_elo(away, artifacts)
    elo_diff = (home_elo - away_elo) if (home_elo and away_elo) else None

    return MatchPrediction(
        home_team=home,
        away_team=away,
        p_home_win=round(p_home_win, 4),
        p_draw=round(p_draw, 4),
        p_away_win=round(p_away_win, 4),
        prediction=prediction,
        home_elo=home_elo,
        away_elo=away_elo,
        elo_diff=round(elo_diff, 1) if elo_diff is not None else None,
        features=feature_row,
        was_inverted=was_inverted,
    )


# =============================================================================
# predict_score — Poisson + Monte Carlo
# =============================================================================

def predict_score(
    home_team: str,
    away_team: str,
    artifacts: MLArtifacts,
    is_neutral: bool = True,
    n_simulations: int = N_SIMULATIONS,
) -> ScorePrediction | None:
    home = resolve_team_name(home_team, artifacts)
    away = resolve_team_name(away_team, artifacts)

    feature_row, was_inverted = _get_features(home, away, artifacts, is_neutral)
    if feature_row is None:
        return None

    home_gf, home_ga, away_gf, away_ga, elo_prob = _extract_goal_features(
        feature_row, was_inverted
    )
    λ_home, λ_away = _compute_lambdas(home_gf, home_ga, away_gf, away_ga, elo_prob)

    rng        = np.random.default_rng()
    home_goals = rng.poisson(λ_home, n_simulations)
    away_goals = rng.poisson(λ_away, n_simulations)

    p_home_win = float((home_goals > away_goals).mean())
    p_draw     = float((home_goals == away_goals).mean())
    p_away_win = float((home_goals < away_goals).mean())

    score_strings = np.char.add(
        np.char.add(home_goals.astype(str), "-"),
        away_goals.astype(str),
    )
    unique_scores, counts = np.unique(score_strings, return_counts=True)
    top_idx = np.argsort(-counts)[:TOP_SCORES]

    top_scores = [
        ScoreEntry(
            score=unique_scores[i],
            probability=round(float(counts[i] / n_simulations), 4),
        )
        for i in top_idx
    ]
    best = top_scores[0].score.split("-")

    return ScorePrediction(
        home_team=home,
        away_team=away,
        predicted_home_goals=int(best[0]),
        predicted_away_goals=int(best[1]),
        expected_home_goals=round(λ_home, 2),
        expected_away_goals=round(λ_away, 2),
        p_home_win=round(p_home_win, 4),
        p_draw=round(p_draw, 4),
        p_away_win=round(p_away_win, 4),
        top_scores=top_scores,
        n_simulations=n_simulations,
    )


# =============================================================================
# Helpers privados
# =============================================================================

def _get_features(
    home_team: str,
    away_team: str,
    artifacts: MLArtifacts,
    is_neutral: bool,
) -> tuple[dict | None, bool]:
    """
    Los nombres ya vienen normalizados desde predict_match / predict_score.
    Ruta A: busca en el fixture pre-computado (directo o invertido).
    Ruta B: construye un vector mínimo con Elo disponible.
    """
    fixture = artifacts.fixture_features

    row = _lookup_fixture(fixture, home_team, away_team)
    if row is not None:
        return _row_to_dict(row), False

    row = _lookup_fixture(fixture, away_team, home_team)
    if row is not None:
        return _row_to_dict(row), True

    # Ruta B
    home_elo_real = get_team_elo(home_team, artifacts)
    away_elo_real = get_team_elo(away_team, artifacts)

    if home_elo_real is None and away_elo_real is None:
        return None, False

    home_elo_val = home_elo_real if home_elo_real is not None else _FALLBACK_ELO
    away_elo_val = away_elo_real if away_elo_real is not None else _FALLBACK_ELO

    elo_diff      = home_elo_val - away_elo_val
    elo_prob_home = 1 / (1 + 10 ** (-elo_diff / 400))

    feature_dict = {col: np.nan for col in MODEL_FEATURES}
    feature_dict["elo_diff"]      = elo_diff
    feature_dict["elo_prob_home"] = elo_prob_home
    feature_dict["is_neutral"]    = int(is_neutral)

    return feature_dict, False


def _lookup_fixture(
    fixture: pd.DataFrame, home: str, away: str
) -> pd.Series | None:
    mask = (fixture["home_team"] == home) & (fixture["away_team"] == away)
    rows = fixture[mask]
    return rows.iloc[0] if len(rows) > 0 else None


def _row_to_dict(row: pd.Series) -> dict:
    return {col: row[col] if col in row.index else np.nan for col in MODEL_FEATURES}


def _extract_goal_features(
    features: dict, was_inverted: bool,
) -> tuple[float | None, float | None, float | None, float | None, float]:
    def _val(key: str) -> float | None:
        v = features.get(key)
        return None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)

    if was_inverted:
        return _val("away_avg_gf"), _val("away_avg_ga"), _val("home_avg_gf"), _val("home_avg_ga"), 1.0 - (_val("elo_prob_home") or 0.5)
    return _val("home_avg_gf"), _val("home_avg_ga"), _val("away_avg_gf"), _val("away_avg_ga"), _val("elo_prob_home") or 0.5


def _compute_lambdas(
    home_gf: float | None, home_ga: float | None,
    away_gf: float | None, away_ga: float | None,
    elo_prob_home: float,
) -> tuple[float, float]:
    adjustment = _ELO_ALPHA * (2 * elo_prob_home - 1)

    if all(v is not None for v in [home_gf, home_ga, away_gf, away_ga]):
        # Base desde stats históricas, corregida por Elo
        λ_home_base = (home_gf + away_ga) / 2  # type: ignore[operator]
        λ_away_base = (away_gf + home_ga) / 2  # type: ignore[operator]
        λ_home = λ_home_base * (1 + adjustment)
        λ_away = λ_away_base * (1 - adjustment)
    else:
        # Solo Elo disponible
        λ_home = _WC_GOALS_PER_TEAM * (1 + adjustment)
        λ_away = _WC_GOALS_PER_TEAM * (1 - adjustment)

    return float(np.clip(λ_home, 0.3, 5.0)), float(np.clip(λ_away, 0.3, 5.0))