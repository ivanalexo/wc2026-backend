"""
Actualización de ELO con la fórmula World Football Elo (eloratings.net),
consistente con la fuente de los ratings base del proyecto.

    We_a   = 1 / (1 + 10^(-(elo_a - elo_b)/400))     # expectativa por ELO
    W_a    = 1 (gana) | 0.5 (empata) | 0 (pierde)    # resultado REAL
    G      = multiplicador por diferencia de goles
    elo_a' = elo_a + K · G · (W_a - We_a)
"""
from __future__ import annotations

from dataclasses import dataclass

WORLD_CUP_K: float = 60.0
_FALLBACK_ELO: float = 1500.0


@dataclass
class FinishedMatch:
    home: str
    away: str
    home_score: int
    away_score: int


def expected_score(elo_a: float, elo_b: float) -> float:
    """Resultado esperado de A (0..1) según la diferencia de ELO."""
    return 1.0 / (1.0 + 10 ** (-(elo_a - elo_b) / 400.0))


def goal_diff_multiplier(goal_diff: int) -> float:
    """
    Multiplicadr G por diferencia de goles (eloratings.net):
      |gd| <= 1 → 1.0
      |gd| == 2 → 1.5
      |gd| >= 3 → (11 + |gd|) / 8
    """
    gd = abs(goal_diff)
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11 + gd) / 8.0


def _result_score(home_score: int, away_score: int) -> tuple[float, float]:
    """(W_home, W_away) ∈ {1, 0.5, 0}."""
    if home_score > away_score:
        return 1.0, 0.0
    if home_score < away_score:
        return 0.0, 1.0
    return 0.5, 0.5


def apply_match(
    elo_home: float,
    elo_away: float,
    home_score: int,
    away_score: int,
    k: float = WORLD_CUP_K,
) -> tuple[float, float]:
    """Devuelve (elo_home', elo_away') tras un partido. Suma cero."""
    we_home = expected_score(elo_home, elo_away)
    w_home, w_away = _result_score(home_score, away_score)
    g = goal_diff_multiplier(home_score - away_score)

    delta_home = k * g * (w_home - we_home)
    return elo_home + delta_home, elo_away - delta_home


def recompute_elo(
    base_elo: dict[str, float],
    finished_matches: list[FinishedMatch],
    k: float = WORLD_CUP_K,
) -> dict[str, float]:
    """
    Recalcula el ELO vigente desde el base + replay cronológico de finalizados.

    `finished_matches` debe venir ordenado por fecha ascendente. Idempotente:
    siempre parte del mismo `base_elo`, así que el resultado solo depende del
    conjunto de partidos jugados, no de cuántas veces se llame.
    """
    elo = dict(base_elo)
    for m in finished_matches:
        elo_h = elo.get(m.home, _FALLBACK_ELO)
        elo_a = elo.get(m.away, _FALLBACK_ELO)
        new_h, new_a = apply_match(elo_h, elo_a, m.home_score, m.away_score, k)
        elo[m.home] = new_h
        elo[m.away] = new_a
    return elo
