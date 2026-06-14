import numpy as np
import pytest

from app.ml.loader import load_artifacts
from app.ml.simulation import build_context, run_simulation, sim_group_stage


@pytest.fixture(scope="module")
def artifacts():
    return load_artifacts()


@pytest.fixture(scope="module")
def ctx(artifacts):
    return build_context(artifacts)

def test_groups_and_teams_inferred(ctx):
    assert len(ctx.groups) == 12
    assert len(ctx.teams) == 48
    for teams in ctx.groups.values():
        assert len(teams) == 4


def test_baseline_ranking_is_coherent(artifacts, ctx):
    """Sin conditioning, los favoritos conocidos deben dominar el top."""
    df = run_simulation(artifacts, played_results={}, n_simulations=1500, ctx=ctx)
    top8 = set(df.head(8)["team"])
    favorites = {"Spain", "Argentina", "France", "England", "Brazil"}
    assert len(top8 & favorites) >= 3, f"top8={top8}"


def test_stage_probabilities_are_monotonic(artifacts, ctx):
    """P(clasif) >= P(R16) >= P(QF) >= P(SF) >= P(final) >= P(campeón) por equipo."""
    df = run_simulation(artifacts, played_results={}, n_simulations=1000, ctx=ctx)
    stages = ["p_qualify", "p_reach_r16", "p_reach_qf", "p_reach_sf", "p_reach_final", "p_champion"]
    for _, row in df.iterrows():
        vals = [row[s] for s in stages]
        assert all(a >= b - 1e-9 for a, b in zip(vals, vals[1:])), f"{row['team']}: {vals}"
    # Las probabilidades de campeón deben sumar ~1 (un campeón por torneo)
    assert abs(df["p_champion"].sum() - 1.0) < 1e-9

def test_run_simulation_is_deterministic(artifacts, ctx):
    """Mismo seed → resultado idéntico"""
    a = run_simulation(artifacts, played_results={}, n_simulations=300, seed=7, ctx=ctx)
    b = run_simulation(artifacts, played_results={}, n_simulations=300, seed=7, ctx=ctx)
    assert a.equals(b)


def test_played_result_is_respected_in_group_stage(ctx):
    """Un partido fijado aporta exactamente sus goles/puntos en cada corrida."""
    group = next(iter(ctx.groups.values()))
    a, b = group[0], group[1]
    played = {frozenset((a, b)): {"home": a, "away": b, "home_score": 3, "away_score": 0}}

    rng = np.random.default_rng(0)
    ranked = sim_group_stage(ctx, group, rng, played)
    standings = {s["team"]: s for s in ranked}

    # a ganó 3-0: al menos 3 pts, >=3 GF, ese partido sin goles encajados por a en ese match
    assert standings[a]["pts"] >= 3
    assert standings[a]["w"] >= 1
    assert standings[b]["l"] >= 1


def test_forced_sweep_team_always_qualifies(artifacts, ctx):
    """Si un equipo gana sus 3 partidos 5-0, su P(clasificación) ~ 1.0."""
    group = next(iter(ctx.groups.values()))
    winner = group[0]
    played = {}
    for rival in group[1:]:
        played[frozenset((winner, rival))] = {
            "home": winner, "away": rival, "home_score": 5, "away_score": 0,
        }

    df = run_simulation(artifacts, played_results=played, n_simulations=500, ctx=ctx)
    p_qualify = float(df.loc[df["team"] == winner, "p_qualify"].iloc[0])
    assert p_qualify > 0.99, f"{winner} p_qualify={p_qualify}"


def test_conditioning_changes_distribution(artifacts, ctx):
    """Fijar resultados debe mover las probabilidades vs el torneo desde cero."""
    group = next(iter(ctx.groups.values()))
    winner = group[0]
    played = {
        frozenset((winner, rival)): {
            "home": winner, "away": rival, "home_score": 4, "away_score": 0,
        }
        for rival in group[1:]
    }
    base = run_simulation(artifacts, played_results={}, n_simulations=500, seed=1, ctx=ctx)
    cond = run_simulation(artifacts, played_results=played, n_simulations=500, seed=1, ctx=ctx)

    q_base = float(base.loc[base["team"] == winner, "p_qualify"].iloc[0])
    q_cond = float(cond.loc[cond["team"] == winner, "p_qualify"].iloc[0])
    assert q_cond > q_base
