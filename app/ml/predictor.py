# =============================================================================
# app/ml/predictor.py — Predicción de resultado y marcador
# =============================================================================
# Expone dos funciones públicas:
#
#   predict_match(home, away, artifacts)
#     → XGBoost con 19 features → P(H), P(D), P(A)
#
#   predict_score(home, away, artifacts)
#     → Distribución de Poisson + Monte Carlo (10,000 simulaciones)
#       → marcador más probable + distribución de resultados exactos
#
# Las dos funciones son INDEPENDIENTES entre sí.
# predict_score no invoca al modelo XGBoost — usa directamente
# las estadísticas de ataque/defensa para estimar los λ de Poisson.
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.ml.loader import MLArtifacts, MODEL_FEATURES, CLASS_ORDER

# --- Constantes -------------------------------------------------------------

# Promedio histórico de goles por equipo por partido en Mundiales (2.82 / 2)
_WC_GOALS_PER_TEAM: float = 1.41

# Factor de ajuste Elo para el fallback de λ cuando no hay stats de ataque/defensa.
# Con α=0.4: si elo_prob_home=0.7 → λ_home=1.63, λ_away=1.18 (total=2.82)
_ELO_ALPHA: float = 0.4

N_SIMULATIONS: int = 10_000
TOP_SCORES: int    = 5


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
    prediction: str           # H | D | A
    home_elo: float | None
    away_elo: float | None
    elo_diff: float | None
    features: dict
    was_inverted: bool


@dataclass
class ScoreEntry:
    """Un marcador posible y su probabilidad estimada."""
    score: str           # "2-1"
    probability: float   # 0.087


@dataclass
class ScorePrediction:
    home_team: str
    away_team: str
    predicted_home_goals: int    # moda de la simulación
    predicted_away_goals: int    # moda de la simulación
    expected_home_goals: float   # λ_home usado
    expected_away_goals: float   # λ_away usado
    p_home_win: float            # P(home > away) en simulación
    p_draw: float                # P(home == away)
    p_away_win: float            # P(away > home)
    top_scores: list[ScoreEntry] # los TOP_SCORES marcadores más probables
    n_simulations: int = N_SIMULATIONS


# =============================================================================
# predict_match — XGBoost
# =============================================================================

def predict_match(
    home_team: str,
    away_team: str,
    artifacts: MLArtifacts,
    is_neutral: bool = True,
) -> MatchPrediction | None:
    """
    Predice el resultado (H/D/A) con el modelo XGBoost.
    Retorna None si no hay señal de Elo para ninguno de los dos equipos.
    """
    feature_row, was_inverted = _get_features(
        home_team, away_team, artifacts, is_neutral
    )
    if feature_row is None:
        return None

    X = pd.DataFrame([feature_row])[MODEL_FEATURES]
    X_imputed = artifacts.imputer.transform(X)

    # predict_proba → [P(A), P(D), P(H)]  (ver CLASS_ORDER en loader.py)
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

    home_elo = artifacts.team_elo.get(home_team)
    away_elo = artifacts.team_elo.get(away_team)
    elo_diff = (home_elo - away_elo) if (home_elo and away_elo) else None

    return MatchPrediction(
        home_team=home_team,
        away_team=away_team,
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
    """
    Predice el marcador más probable usando distribución de Poisson
    simulada con Monte Carlo.

    Modelo de goles esperados:
      Si hay stats de ataque/defensa en los features pre-computados:
        λ_home = (home_avg_gf + away_avg_ga) / 2
        λ_away = (away_avg_gf + home_avg_ga) / 2
      Si solo hay Elo disponible (Ruta B):
        λ_home = WC_AVG * (1 + α * (2*elo_prob - 1))
        λ_away = WC_AVG * (1 - α * (2*elo_prob - 1))

    Retorna None si no hay señal de Elo para ninguno de los dos equipos.
    """
    feature_row, was_inverted = _get_features(
        home_team, away_team, artifacts, is_neutral
    )
    if feature_row is None:
        return None

    # Extraemos stats desde la perspectiva CORRECTA del request.
    # Si was_inverted, los features están en orden (away, home) del request,
    # así que swapeamos las columnas home/away antes de calcular λ.
    home_gf, home_ga, away_gf, away_ga, elo_prob = _extract_goal_features(
        feature_row, was_inverted
    )

    λ_home, λ_away = _compute_lambdas(home_gf, home_ga, away_gf, away_ga, elo_prob)

    # --- Simulación Monte Carlo ---------------------------------------------
    rng        = np.random.default_rng()
    home_goals = rng.poisson(λ_home, n_simulations)
    away_goals = rng.poisson(λ_away, n_simulations)

    p_home_win = float((home_goals > away_goals).mean())
    p_draw     = float((home_goals == away_goals).mean())
    p_away_win = float((home_goals < away_goals).mean())

    # --- Distribución de marcadores -----------------------------------------
    # Construimos un array de strings "h-a" y contamos frecuencias
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

    # Marcador más probable (moda)
    best = top_scores[0].score.split("-")
    predicted_home = int(best[0])
    predicted_away = int(best[1])

    return ScorePrediction(
        home_team=home_team,
        away_team=away_team,
        predicted_home_goals=predicted_home,
        predicted_away_goals=predicted_away,
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
    Ruta A: busca el par en el fixture pre-computado (directo o invertido).
    Ruta B: construye un vector mínimo con solo Elo si no está en el fixture.
    Retorna (feature_dict, was_inverted).
    """
    fixture = artifacts.fixture_features

    row = _lookup_fixture(fixture, home_team, away_team)
    if row is not None:
        return _row_to_dict(row), False

    row = _lookup_fixture(fixture, away_team, home_team)
    if row is not None:
        return _row_to_dict(row), True

    # Ruta B
    home_elo = artifacts.team_elo.get(home_team)
    away_elo = artifacts.team_elo.get(away_team)

    if home_elo is None and away_elo is None:
        return None, False

    elo_diff      = (home_elo or 0) - (away_elo or 0)
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
    features: dict,
    was_inverted: bool,
) -> tuple[float | None, float | None, float | None, float | None, float]:
    """
    Extrae las 4 stats de goles y elo_prob desde la perspectiva correcta del request.
    Si was_inverted, intercambia las columnas home/away.
    """
    def _val(key: str) -> float | None:
        v = features.get(key)
        return None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)

    if was_inverted:
        home_gf    = _val("away_avg_gf")
        home_ga    = _val("away_avg_ga")
        away_gf    = _val("home_avg_gf")
        away_ga    = _val("home_avg_ga")
        elo_prob   = 1.0 - (_val("elo_prob_home") or 0.5)
    else:
        home_gf    = _val("home_avg_gf")
        home_ga    = _val("home_avg_ga")
        away_gf    = _val("away_avg_gf")
        away_ga    = _val("away_avg_ga")
        elo_prob   = _val("elo_prob_home") or 0.5

    return home_gf, home_ga, away_gf, away_ga, elo_prob


def _compute_lambdas(
    home_gf: float | None,
    home_ga: float | None,
    away_gf: float | None,
    away_ga: float | None,
    elo_prob_home: float,
) -> tuple[float, float]:
    """
    Calcula los parámetros λ de Poisson para home y away.

    Con stats disponibles (Ruta A):
      λ_home = (home_avg_gf + away_avg_ga) / 2
      λ_away = (away_avg_gf + home_avg_ga) / 2
      → Promedia la capacidad ofensiva del equipo con la vulnerabilidad defensiva del rival.

    Sin stats (Ruta B, solo Elo):
      λ_home = WC_AVG * (1 + α * (2*p - 1))
      λ_away = WC_AVG * (1 - α * (2*p - 1))
      → Distribuye el promedio histórico de goles según la ventaja Elo.
      → La suma siempre es WC_GOALS_PER_TEAM * 2 = 2.82.
    """
    has_stats = all(v is not None for v in [home_gf, home_ga, away_gf, away_ga])

    if has_stats:
        λ_home = (home_gf + away_ga) / 2   # type: ignore[operator]
        λ_away = (away_gf + home_ga) / 2   # type: ignore[operator]
    else:
        adjustment = _ELO_ALPHA * (2 * elo_prob_home - 1)
        λ_home = _WC_GOALS_PER_TEAM * (1 + adjustment)
        λ_away = _WC_GOALS_PER_TEAM * (1 - adjustment)

    # Clamp: valores extremos arruinan la distribución Poisson
    λ_home = float(np.clip(λ_home, 0.3, 5.0))
    λ_away = float(np.clip(λ_away, 0.3, 5.0))

    return λ_home, λ_away