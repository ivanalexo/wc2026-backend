#!/usr/bin/env python3
"""
Re-corre la simulación Monte Carlo condicionada a los resultados de la DB y la
persiste en la tabla `simulation_results` (lo mismo que dispara /admin/sync).

Usage:
    python scripts/run_simulation.py                 # regenera y persiste en DB
    python scripts/run_simulation.py --n 2000        # menos iteraciones
    python scripts/run_simulation.py --compare       # top-10 base vs condicionado
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db.session import SessionLocal
from app.ml.elo import FinishedMatch, recompute_elo
from app.ml.loader import load_artifacts
from app.ml.simulation import base_team_elo, build_context, run_simulation
from app.services.simulation_runner import (
    _finished_matches,
    build_played_results,
    regenerate_simulation,
)


def _print_top(title: str, df) -> None:
    print(f"\n  {title}")
    print(f"  {'#':>2}  {'Equipo':<22} {'P(Clasif)':>10} {'P(Campeón)':>11}")
    print(f"  {'-'*48}")
    for i, row in df.head(10).iterrows():
        print(f"  {i+1:>2}  {row['team']:<22} {row['p_qualify']:>9.1%} {row['p_champion']:>10.2%}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-simulación Monte Carlo WC2026")
    parser.add_argument("--n", type=int, default=10_000, help="Nº de simulaciones")
    parser.add_argument("--compare", action="store_true", help="Top-10 base vs condicionado (no escribe en db)")
    args = parser.parse_args()

    print("Cargando artifacts...")
    artifacts = load_artifacts()
    db = SessionLocal()
    try:
        finished = _finished_matches(db)
        print(f"Partidos finalizados en la DB: {len(finished)}")

        if args.compare:
            played = build_played_results(finished)
            base_elo = base_team_elo(artifacts)
            current_elo = recompute_elo(
                base_elo,
                [FinishedMatch(m.home_team, m.away_team, int(m.home_score), int(m.away_score))
                 for m in finished],
            )
            print(f"\nSimulando (n={args.n})...")
            df_base = run_simulation(artifacts, played_results={}, n_simulations=args.n)
            ctx = build_context(artifacts, elo_override=current_elo)
            df_cond = run_simulation(artifacts, played_results=played, n_simulations=args.n, ctx=ctx)
            _print_top("BASE (sin conditioning, ELO pre-torneo):", df_base)
            _print_top("CONDICIONADO (resultados reales + ELO actualizado):", df_cond)
            return

        result = regenerate_simulation(db, artifacts, n_simulations=args.n)
    finally:
        db.close()

    _print_top("Persistido en simulation_results:", pd.DataFrame(result["top_10"]))
    print(f"\n  Partidos jugados: {result['played_matches']} | n={result['n_simulations']}")


if __name__ == "__main__":
    main()
