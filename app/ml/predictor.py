from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.ml.loader import MLArtifacts, MODEL_FEATURES, CLASS_ORDER

_FALLBACK_ELO: float = 1500.0
_WC_GOALS_PER_TEAM: float = 1.41
_MAX_GOALS_ANALYTIC: int = 15   # techo para la suma exacta de Poisson
N_SIMULATIONS: int = 10_000
TOP_SCORES: int = 5

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

def resolve_team_name(name: str, artifacts: MLArtifacts) -> str:
    """Convierte cualquier variante al nombre canónico del fixture."""
    return artifacts.team_name_map.get(name.strip().lower(), name.strip())


def get_team_elo(name: str, artifacts: MLArtifacts) -> float | None:
    """Busca el Elo de un equipo de forma case-insensitive."""
    return artifacts.team_elo.get(name.strip().lower())

def predict_match(
    home_team: str,
    away_team: str,
    artifacts: MLArtifacts,
    is_neutral: bool = True,
) -> MatchPrediction | None:
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

    match_pred = predict_match(home, away, artifacts, is_neutral)
    if match_pred is None:
        return None

    home_gf, home_ga, away_gf, away_ga = _extract_goal_stats(
        feature_row, was_inverted, home, away, artifacts
    )
    λ_home_base, λ_away_base = _compute_base_lambdas(home_gf, home_ga, away_gf, away_ga)

    λ_home, λ_away = _calibrate_lambdas(
        λ_home_base, λ_away_base,
        match_pred.p_home_win, match_pred.p_draw, match_pred.p_away_win,
    )

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

def _poisson_win_probs(
    λ_home: float,
    λ_away: float,
    max_goals: int = _MAX_GOALS_ANALYTIC,
) -> tuple[float, float, float]:
    """
    Calcula P(home win), P(draw), P(away win) de forma exacta
    a partir de las PMF de Poisson.
    """
    exp_h = math.exp(-λ_home)
    exp_a = math.exp(-λ_away)

    pmf_h = [exp_h * (λ_home ** k) / math.factorial(k) for k in range(max_goals + 1)]
    pmf_a = [exp_a * (λ_away ** k) / math.factorial(k) for k in range(max_goals + 1)]

    p_home_win = p_draw = p_away_win = 0.0
    for h, ph in enumerate(pmf_h):
        for a, pa in enumerate(pmf_a):
            prob = ph * pa
            if   h > a: p_home_win += prob
            elif h == a: p_draw    += prob
            else:        p_away_win += prob

    return p_home_win, p_draw, p_away_win


def _calibrate_lambdas(
    λ_home_base: float,
    λ_away_base: float,
    target_p_home: float,
    target_p_draw: float,
    target_p_away: float,
    max_iter: int = 64,
    tol: float = 1e-6,
) -> tuple[float, float]:

    λ_total = λ_home_base + λ_away_base

    # Empate dominante → lambdas iguales
    if target_p_draw >= target_p_home and target_p_draw >= target_p_away:
        half = float(np.clip(λ_total / 2.0, 0.3, 5.0))
        return half, half

    # Bisección sobre la fracción de goles del local
    lo, hi = 0.01, 0.99
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        p_hw, _, _ = _poisson_win_probs(λ_total * mid, λ_total * (1.0 - mid))
        if abs(p_hw - target_p_home) < tol:
            break
        if p_hw < target_p_home:
            lo = mid
        else:
            hi = mid

    frac  = (lo + hi) / 2.0
    λ_home = float(np.clip(λ_total * frac,          0.3, 5.0))
    λ_away = float(np.clip(λ_total * (1.0 - frac),  0.3, 5.0))
    return λ_home, λ_away

def _extract_goal_stats(
    features: dict,
    was_inverted: bool,
    home: str,
    away: str,
    artifacts: MLArtifacts,
) -> tuple[float | None, float | None, float | None, float | None]:
    """
    Devuelve (home_gf, home_ga, away_gf, away_ga).

    Prioridad:
    1. team_goals (del dataset de goalscorers).
    2. Promedios pre-computados del fixture (home_avg_gf, etc.).
    """
    def _feat(key: str) -> float | None:
        v = features.get(key)
        return None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)

    team_goals: dict = getattr(artifacts, "team_goals", {}) or {}

    if team_goals:
        home_key = home.strip().lower()
        away_key = away.strip().lower()

        home_stats = team_goals.get(home_key)
        away_stats = team_goals.get(away_key)

        home_gf = home_stats.avg_gf if home_stats else None
        home_ga = home_stats.avg_ga if home_stats else None
        away_gf = away_stats.avg_gf if away_stats else None
        away_ga = away_stats.avg_ga if away_stats else None

        if all(v is not None for v in [home_gf, home_ga, away_gf, away_ga]):
            if was_inverted:
                return away_gf, away_ga, home_gf, home_ga  # type: ignore[return-value]
            return home_gf, home_ga, away_gf, away_ga      # type: ignore[return-value]

    if was_inverted:
        return _feat("away_avg_gf"), _feat("away_avg_ga"), _feat("home_avg_gf"), _feat("home_avg_ga")
    return _feat("home_avg_gf"), _feat("home_avg_ga"), _feat("away_avg_gf"), _feat("away_avg_ga")


def _compute_base_lambdas(
    home_gf: float | None,
    home_ga: float | None,
    away_gf: float | None,
    away_ga: float | None,
) -> tuple[float, float]:
    """
    Lambdas base a partir de promedios de goles.
    No incluye ajuste ELO
    """
    if all(v is not None for v in [home_gf, home_ga, away_gf, away_ga]):
        λ_home = (home_gf + away_ga) / 2   # type: ignore[operator]
        λ_away = (away_gf + home_ga) / 2   # type: ignore[operator]
    else:
        λ_home = _WC_GOALS_PER_TEAM
        λ_away = _WC_GOALS_PER_TEAM

    return float(np.clip(λ_home, 0.3, 5.0)), float(np.clip(λ_away, 0.3, 5.0))

def _get_features(
    home_team: str,
    away_team: str,
    artifacts: MLArtifacts,
    is_neutral: bool,
) -> tuple[dict | None, bool]:
    fixture = artifacts.fixture_features

    row = _lookup_fixture(fixture, home_team, away_team)
    if row is not None:
        return _row_to_dict(row), False

    row = _lookup_fixture(fixture, away_team, home_team)
    if row is not None:
        return _row_to_dict(row), True

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


def _lookup_fixture(fixture: pd.DataFrame, home: str, away: str) -> pd.Series | None:
    mask = (fixture["home_team"] == home) & (fixture["away_team"] == away)
    rows = fixture[mask]
    return rows.iloc[0] if len(rows) > 0 else None


def _row_to_dict(row: pd.Series) -> dict:
    return {col: row[col] if col in row.index else np.nan for col in MODEL_FEATURES}