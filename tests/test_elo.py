import math

from app.ml.elo import (
    FinishedMatch,
    apply_match,
    expected_score,
    goal_diff_multiplier,
    recompute_elo,
)


def test_expected_score_symmetry():
    assert expected_score(1500, 1500) == 0.5
    assert math.isclose(expected_score(1900, 1700) + expected_score(1700, 1900), 1.0)
    assert expected_score(1900, 1700) > 0.5


def test_goal_diff_multiplier():
    assert goal_diff_multiplier(0) == 1.0
    assert goal_diff_multiplier(1) == 1.0
    assert goal_diff_multiplier(-1) == 1.0
    assert goal_diff_multiplier(2) == 1.5
    assert goal_diff_multiplier(3) == (11 + 3) / 8.0
    assert goal_diff_multiplier(-4) == (11 + 4) / 8.0


def test_apply_match_is_zero_sum():
    """Lo que gana un equipo lo pierde el otro."""
    eh, ea = apply_match(1900, 1700, 2, 0)
    assert math.isclose((eh - 1900) + (ea - 1700), 0.0, abs_tol=1e-9)


def test_draw_favorite_loses_underdog_gains():
    """Caso del usuario: el modelo esperaba ganador pero empataron."""
    eh, ea = apply_match(1900, 1700, 1, 1)   # favorito local empata
    assert eh < 1900   # favorito pierde ELO
    assert ea > 1700   # menos favorito gana ELO
    # ~16 puntos con K=60, We≈0.76
    assert math.isclose(1900 - eh, 15.6, abs_tol=0.3)


def test_upset_swings_more_than_expected_win():
    """Que gane el menos favorito mueve más el ELO que si gana el favorito."""
    # Favorito gana 1-0
    fav_h, _ = apply_match(1900, 1700, 1, 0)
    gain_expected = fav_h - 1900
    # Menos favorito gana 1-0 (sorpresa)
    _, und_a = apply_match(1900, 1700, 0, 1)
    gain_upset = und_a - 1700
    assert gain_upset > gain_expected


def test_recompute_replays_and_accumulates():
    base = {"A": 1800.0, "B": 1600.0, "C": 1700.0}
    matches = [
        FinishedMatch("A", "B", 2, 0),   # A favorito gana → sube poco
        FinishedMatch("C", "A", 1, 0),   # C vence a A → A baja
    ]
    elo = recompute_elo(base, matches)
    assert elo["A"] != base["A"]
    assert elo["B"] < base["B"]          # B perdió contra favorito
    assert elo["C"] > base["C"]          # C dio el golpe


def test_recompute_is_idempotent():
    """Llamar dos veces desde el mismo base da el mismo resultado."""
    base = {"A": 1800.0, "B": 1600.0}
    matches = [FinishedMatch("A", "B", 3, 1)]
    a = recompute_elo(base, matches)
    b = recompute_elo(base, matches)
    assert a == b
    # Y no muta el base original
    assert base == {"A": 1800.0, "B": 1600.0}


def test_recompute_empty_returns_base():
    base = {"A": 1800.0, "B": 1600.0}
    assert recompute_elo(base, []) == base
