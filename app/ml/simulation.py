"""
Simulación Monte Carlo del Mundial 2026).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.ml.loader import MLArtifacts, MODEL_FEATURES

N_SIMULATIONS_DEFAULT = 10_000
DEFAULT_SEED = 42

STAGE_KEYS = ["qualified", "r16", "qf", "sf", "final", "champion"]

R32_SLOTS = [
    (73, "2A",  "2B",  None),
    (74, "1E",  "3rd", "ABCDF"),
    (75, "1F",  "2C",  None),
    (76, "1C",  "2F",  None),
    (77, "1I",  "3rd", "CDFGH"),
    (78, "2E",  "2I",  None),
    (79, "1A",  "3rd", "CEFHI"),
    (80, "1L",  "3rd", "EHIJK"),
    (81, "1D",  "3rd", "BEFIJ"),
    (82, "1G",  "3rd", "AEHIJ"),
    (83, "2K",  "2L",  None),
    (84, "1H",  "2J",  None),
    (85, "1B",  "3rd", "EFGIJ"),
    (86, "1J",  "2H",  None),
    (87, "1K",  "3rd", "DEIJL"),
    (88, "2D",  "2G",  None),
]
R16_PAIRS = [(1, 4), (0, 2), (3, 5), (6, 7), (10, 11), (8, 9), (13, 15), (12, 14)]
QF_PAIRS = [(0, 1), (4, 5), (2, 3), (6, 7)]
SF_PAIRS = [(0, 1), (2, 3)]

@dataclass
class SimulationContext:
    groups: dict[str, list[str]]
    team_features: dict[str, dict]
    proba_cache: dict[tuple[str, str], np.ndarray]
    teams: list[str]


def _infer_groups(fixture: pd.DataFrame) -> dict[str, list[str]]:
    """Infiere los 12 grupos (cada equipo enfrenta a sus 3 rivales)."""
    from collections import defaultdict

    adj: dict[str, set[str]] = defaultdict(set)
    for _, r in fixture.iterrows():
        adj[r.home_team].add(r.away_team)
        adj[r.away_team].add(r.home_team)

    groups: dict[str, list[str]] = {}
    seen: set[str] = set()
    group_letter = 65  # ord('A')

    for t in list(adj.keys()):
        if t in seen:
            continue
        cands = [t] + list(adj[t])
        for combo in itertools.combinations(cands, 4):
            if all(b in adj[a] for a, b in itertools.combinations(combo, 2)):
                groups[chr(group_letter)] = sorted(combo)
                seen.update(combo)
                group_letter += 1
                break
    return groups


def _build_team_features(fixture: pd.DataFrame) -> dict[str, dict]:
    """Tabla de features por equipo (para generar partidos de eliminatoria)."""
    team_features: dict[str, dict] = {}
    for _, row in fixture.iterrows():
        for side, team in [("home", row.home_team), ("away", row.away_team)]:
            if team not in team_features:
                team_features[team] = {
                    "elo":          row.home_elo if side == "home" else row.away_elo,
                    "avg_gf":       row[f"{side}_avg_gf"],
                    "avg_ga":       row[f"{side}_avg_ga"],
                    "form_5":       row[f"{side}_form_5"],
                    "form_10":      row[f"{side}_form_10"],
                    "penalty_rate": row[f"{side}_penalty_rate"],
                }
    return team_features


def _build_feature_row(home: str, away: str, team_features: dict[str, dict]) -> dict:
    """Construye features para un partido de eliminatoria entre dos equipos."""
    tf_h = team_features.get(home, {})
    tf_a = team_features.get(away, {})
    elo_h = tf_h.get("elo", 1500)
    elo_a = tf_a.get("elo", 1500)
    elo_diff = elo_h - elo_a
    elo_prob_h = 1 / (1 + 10 ** (-elo_diff / 400))
    return {
        "elo_diff":          elo_diff,
        "elo_prob_home":     elo_prob_h,
        "is_neutral":        1,
        "home_form_5":       tf_h.get("form_5",  0.5),
        "away_form_5":       tf_a.get("form_5",  0.5),
        "form_diff_5":       tf_h.get("form_5",  0.5) - tf_a.get("form_5",  0.5),
        "home_form_10":      tf_h.get("form_10", 0.5),
        "away_form_10":      tf_a.get("form_10", 0.5),
        "form_diff_10":      tf_h.get("form_10", 0.5) - tf_a.get("form_10", 0.5),
        "home_avg_gf":       tf_h.get("avg_gf",  1.5),
        "home_avg_ga":       tf_h.get("avg_ga",  1.0),
        "away_avg_gf":       tf_a.get("avg_gf",  1.5),
        "away_avg_ga":       tf_a.get("avg_ga",  1.0),
        "gf_diff":           tf_h.get("avg_gf",  1.5) - tf_a.get("avg_gf",  1.5),
        "ga_diff":           tf_h.get("avg_ga",  1.0) - tf_a.get("avg_ga",  1.0),
        "h2h_total":         5.0,
        "h2h_home_rate":     0.4,
        "home_penalty_rate": tf_h.get("penalty_rate", 0.5),
        "away_penalty_rate": tf_a.get("penalty_rate", 0.5),
    }


def _build_proba_cache(
    artifacts: MLArtifacts,
    team_features: dict[str, dict],
) -> dict[tuple[str, str], np.ndarray]:
    """
    Pre-computa [P(H), P(D), P(A)] para todos los pares de equipos (O(1) en sim).

    - Partidos de grupos: usa las filas ricas de `fixture_features` (h2h/form reales),
      pero recomputa `elo_diff`/`elo_prob_home` desde el ELO de `team_features`. Esto
      hace que un ELO actualizado se refleje en los partidos pendientes; sin override
      es idéntico al base (mismo ELO → mismas columnas).
    - Pares de eliminatoria: usa `_build_feature_row` (genéricos), como el notebook.
    """
    fixture = artifacts.fixture_features
    model = artifacts.model
    imputer = artifacts.imputer

    cache: dict[tuple[str, str], np.ndarray] = {}

    # Partidos de grupos (batch sobre el fixture)
    X_group = fixture[MODEL_FEATURES].copy()
    home_elo = fixture["home_team"].map(lambda t: team_features.get(t, {}).get("elo"))
    away_elo = fixture["away_team"].map(lambda t: team_features.get(t, {}).get("elo"))
    elo_diff = home_elo - away_elo
    X_group["elo_diff"] = elo_diff.to_numpy()
    X_group["elo_prob_home"] = 1.0 / (1.0 + 10 ** (-elo_diff / 400.0)).to_numpy()
    X_group_imp = imputer.transform(X_group)
    group_probas = model.predict_proba(X_group_imp)
    for (_, row), p in zip(fixture.iterrows(), group_probas):
        cache[(row.home_team, row.away_team)] = p

    # Resto de pares posibles (eliminatoria)
    all_teams = list(team_features.keys())
    extra_rows, extra_keys = [], []
    for t1, t2 in itertools.combinations(all_teams, 2):
        if (t1, t2) not in cache and (t2, t1) not in cache:
            extra_rows.append(_build_feature_row(t1, t2, team_features))
            extra_keys.append((t1, t2))

    if extra_rows:
        df_extra = pd.DataFrame(extra_rows)[MODEL_FEATURES]
        df_extra_imp = imputer.transform(df_extra)
        extra_probas = model.predict_proba(df_extra_imp)
        for (t1, t2), p in zip(extra_keys, extra_probas):
            cache[(t1, t2)] = p

    return cache


def base_team_elo(artifacts: MLArtifacts) -> dict[str, float]:
    """ELO base congelado por equipo (del artifact), punto de partida del replay."""
    tf = _build_team_features(artifacts.fixture_features)
    return {team: feats["elo"] for team, feats in tf.items() if feats.get("elo") is not None}


def build_context(
    artifacts: MLArtifacts,
    elo_override: dict[str, float] | None = None,
) -> SimulationContext:
    """
    Construye el contexto reutilizable de simulación a partir de los artifacts.
    """
    fixture = artifacts.fixture_features
    team_features = _build_team_features(fixture)
    if elo_override:
        for team, tf in team_features.items():
            if team in elo_override:
                tf["elo"] = elo_override[team]
    groups = _infer_groups(fixture)
    proba_cache = _build_proba_cache(artifacts, team_features)
    return SimulationContext(
        groups=groups,
        team_features=team_features,
        proba_cache=proba_cache,
        teams=list(team_features.keys()),
    )

def sim_match_proba(ctx: SimulationContext, home: str, away: str) -> np.ndarray:
    """Retorna [P(H), P(D), P(A)] desde caché (invierte si está al revés)."""
    if (home, away) in ctx.proba_cache:
        return ctx.proba_cache[(home, away)]
    if (away, home) in ctx.proba_cache:
        p = ctx.proba_cache[(away, home)]
        return np.array([p[2], p[1], p[0]])
    # Fallback (no debería ocurrir tras pre computo)
    return np.array([1 / 3, 1 / 3, 1 / 3])


def sample_outcome(proba: np.ndarray, rng) -> str:
    """Muestra H/D/A."""
    r = rng.random()
    if r < proba[0]:
        return "H"
    elif r < proba[0] + proba[1]:
        return "D"
    return "A"


def sim_goals(ctx: SimulationContext, home: str, away: str, outcome: str, rng) -> tuple[int, int]:
    """Simula marcador coherente con el outcome, sin rejection sampling."""
    lam_h = max(0.4, ctx.team_features.get(home, {}).get("avg_gf", 1.5) * 0.65)
    lam_a = max(0.4, ctx.team_features.get(away, {}).get("avg_gf", 1.5) * 0.65)

    if outcome == "D":
        g = min(rng.poisson((lam_h + lam_a) / 2), 4)
        return g, g
    elif outcome == "H":
        g_h = rng.poisson(lam_h)
        g_a = rng.poisson(lam_a)
        if g_h <= g_a:
            g_h = g_a + 1
        return min(g_h, 6), min(g_a, 5)
    else:  # "A"
        g_h = rng.poisson(lam_h)
        g_a = rng.poisson(lam_a)
        if g_a <= g_h:
            g_a = g_h + 1
        return min(g_h, 5), min(g_a, 6)


def sim_penalty_shootout(ctx: SimulationContext, home: str, away: str, rng) -> str:
    """Penales: retorna 'H' o 'A' según penalty_rate de cada equipo."""
    pr_h = ctx.team_features.get(home, {}).get("penalty_rate", 0.5)
    pr_a = ctx.team_features.get(away, {}).get("penalty_rate", 0.5)
    p_home = pr_h / (pr_h + pr_a + 1e-9)
    return "H" if rng.random() < p_home else "A"


def sim_knockout_match(ctx: SimulationContext, home: str, away: str, rng) -> str:
    """Partido de eliminatoria (sin empate): retorna el equipo ganador."""
    proba = sim_match_proba(ctx, home, away)
    outcome = sample_outcome(proba, rng)
    if outcome == "H":
        return home
    elif outcome == "A":
        return away
    pen_winner = sim_penalty_shootout(ctx, home, away, rng)
    return home if pen_winner == "H" else away

def _oriented_goals(
    played_results: dict[frozenset, dict],
    a: str,
    b: str,
) -> tuple[int, int] | None:
    """
    Si el par {a, b} ya se jugo, devuelve (goles_de_a, goles_de_b) respetando
    la orientacion con que se procesa el partido en la simulacion.
    """
    rec = played_results.get(frozenset((a, b)))
    if rec is None:
        return None
    if rec["home"] == a:
        return rec["home_score"], rec["away_score"]
    return rec["away_score"], rec["home_score"]

def sim_group_stage(
    ctx: SimulationContext,
    group_teams: list[str],
    rng,
    played_results: dict[frozenset, dict],
) -> list[dict]:
    """
    Simula los 6 partidos de un grupo. Los partidos ya jugados se
    fijan con su marcador real; el resto se muestrea. Ordena por pts → gd → gf → nombre.
    """
    standings = {t: {"team": t, "pts": 0, "gf": 0, "ga": 0, "gd": 0,
                     "w": 0, "d": 0, "l": 0} for t in group_teams}

    for home, away in itertools.combinations(group_teams, 2):
        real = _oriented_goals(played_results, home, away)
        if real is not None:
            g_h, g_a = real
            result = "H" if g_h > g_a else ("A" if g_a > g_h else "D")
        else:
            proba = sim_match_proba(ctx, home, away)
            result = sample_outcome(proba, rng)
            g_h, g_a = sim_goals(ctx, home, away, result, rng)

        standings[home]["gf"] += g_h
        standings[home]["ga"] += g_a
        standings[away]["gf"] += g_a
        standings[away]["ga"] += g_h

        if result == "H":
            standings[home]["pts"] += 3
            standings[home]["w"] += 1
            standings[away]["l"] += 1
        elif result == "D":
            standings[home]["pts"] += 1
            standings[away]["pts"] += 1
            standings[home]["d"] += 1
            standings[away]["d"] += 1
        else:
            standings[away]["pts"] += 3
            standings[away]["w"] += 1
            standings[home]["l"] += 1

    for t in standings:
        standings[t]["gd"] = standings[t]["gf"] - standings[t]["ga"]

    return sorted(
        standings.values(),
        key=lambda s: (-s["pts"], -s["gd"], -s["gf"], s["team"]),
    )


def assign_third_place_teams(all_thirds_list: list, rng) -> dict:
    """Asigna los 8 mejores terceros a los slots R32 respetando elegibilidad FIFA."""
    sorted_thirds = sorted(
        all_thirds_list,
        key=lambda t: (-t["pts"], -t["gd"], -t["gf"], t["team"]),
    )
    third_slots = {m: elig for m, _, _, elig in R32_SLOTS if elig is not None}
    qualified_groups = {t["group"] for t in sorted_thirds[:8]}
    eligible_by_slot = {
        m: [t for t in sorted_thirds[:8] if t["group"] in elig and t["group"] in qualified_groups]
        for m, elig in third_slots.items()
    }
    slots_ordered = sorted(third_slots.keys(), key=lambda m: len(eligible_by_slot[m]))

    def backtrack(slot_idx, used_groups, assignment):
        if slot_idx == len(slots_ordered):
            return assignment
        match_id = slots_ordered[slot_idx]
        for candidate in eligible_by_slot[match_id]:
            if candidate["group"] in used_groups:
                continue
            used_groups.add(candidate["group"])
            assignment[match_id] = candidate["team"]
            result = backtrack(slot_idx + 1, used_groups, assignment)
            if result is not None:
                return result
            used_groups.discard(candidate["group"])
            del assignment[match_id]
        return None

    result = backtrack(0, set(), {})
    if result is not None:
        return result

    assignment, used_groups = {}, set()
    for match_id in slots_ordered:
        for candidate in sorted_thirds:
            if candidate["group"] not in used_groups:
                assignment[match_id] = candidate["team"]
                used_groups.add(candidate["group"])
                break
    return assignment


def simulate_tournament(
    ctx: SimulationContext,
    rng,
    played_results: dict[frozenset, dict],
) -> dict[str, set]:
    """Simula un torneo completo. Retorna {stage: set_de_equipos_que_alcanzaron}."""
    reached = {k: set() for k in STAGE_KEYS}

    all_firsts, all_seconds, all_thirds_list = {}, {}, []
    for grp_letter, grp_teams in ctx.groups.items():
        ranked = sim_group_stage(ctx, grp_teams, rng, played_results)
        all_firsts[grp_letter] = ranked[0]
        all_seconds[grp_letter] = ranked[1]
        third = ranked[2]
        third["group"] = grp_letter
        all_thirds_list.append(third)

    for g in ctx.groups:
        reached["qualified"].add(all_firsts[g]["team"])
        reached["qualified"].add(all_seconds[g]["team"])

    third_assignment = assign_third_place_teams(all_thirds_list, rng)
    for team in third_assignment.values():
        reached["qualified"].add(team)

    def resolve_slot(slot_code, match_id):
        if slot_code.startswith("1"):
            return all_firsts[slot_code[1]]["team"]
        elif slot_code.startswith("2"):
            return all_seconds[slot_code[1]]["team"]
        return third_assignment[match_id]

    r32_participants = [
        (resolve_slot(s1, mid), resolve_slot(s2, mid))
        for mid, s1, s2, _ in R32_SLOTS
    ]

    r32_winners = []
    for home, away in r32_participants:
        winner = sim_knockout_match(ctx, home, away, rng)
        r32_winners.append(winner)
        reached["r16"].add(winner)

    r16_winners = []
    for ia, ib in R16_PAIRS:
        winner = sim_knockout_match(ctx, r32_winners[ia], r32_winners[ib], rng)
        r16_winners.append(winner)
        reached["qf"].add(winner)

    qf_winners = []
    for ia, ib in QF_PAIRS:
        winner = sim_knockout_match(ctx, r16_winners[ia], r16_winners[ib], rng)
        qf_winners.append(winner)
        reached["sf"].add(winner)

    sf_winners = []
    for ia, ib in SF_PAIRS:
        winner = sim_knockout_match(ctx, qf_winners[ia], qf_winners[ib], rng)
        sf_winners.append(winner)
        reached["final"].add(winner)

    champion = sim_knockout_match(ctx, sf_winners[0], sf_winners[1], rng)
    reached["champion"].add(champion)

    return reached


#  corre N torneos y agrega probabilidades
def run_simulation(
    artifacts: MLArtifacts,
    played_results: dict[frozenset, dict] | None = None,
    n_simulations: int = N_SIMULATIONS_DEFAULT,
    seed: int = DEFAULT_SEED,
    ctx: SimulationContext | None = None,
) -> pd.DataFrame:
    """
    Corre `n_simulations` torneos condicionados a `played_results` y devuelve un
    DataFrame con columnas: team, elo, p_qualify, p_reach_r16, p_reach_qf,
    p_reach_sf, p_reach_final, p_champion (ordenado por p_champion desc).

    Funcin pura: no escribe archivos ni toca la DB.
    """
    played_results = played_results or {}
    if ctx is None:
        ctx = build_context(artifacts)

    stage_counts = {t: np.zeros(6, dtype=int) for t in ctx.teams}

    master = np.random.default_rng(seed)
    for _ in range(n_simulations):
        sim_rng = np.random.default_rng(master.integers(0, 2**31))
        reached = simulate_tournament(ctx, sim_rng, played_results)
        for stage_i, stage_key in enumerate(STAGE_KEYS):
            for team in reached[stage_key]:
                stage_counts[team][stage_i] += 1

    cols = ["p_qualify", "p_reach_r16", "p_reach_qf", "p_reach_sf", "p_reach_final", "p_champion"]
    df = pd.DataFrame(
        {t: stage_counts[t] / n_simulations for t in ctx.teams},
        index=cols,
    ).T.reset_index().rename(columns={"index": "team"})
    df["elo"] = df["team"].map(lambda t: ctx.team_features[t]["elo"])
    df = df[["team", "elo"] + cols]
    return df.sort_values("p_champion", ascending=False).reset_index(drop=True)
