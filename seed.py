# =============================================================================
# seed.py — Poblar la base de datos con equipos y partidos
# =============================================================================
# Ejecutar UNA sola vez (o cuando quieras resetear los datos):
#   python seed.py
#
# Lee los archivos de artifacts/ y popula las tablas teams y matches.
# Es idempotente: si el equipo o partido ya existe, lo omite.
# =============================================================================

import sys
from pathlib import Path

import pandas as pd

# Aseguramos que Python encuentre el paquete app
sys.path.insert(0, str(Path(__file__).parent))

from app.db.base import Base
from app.db.models.match import Match
from app.db.models.team import Team
from app.db.session import SessionLocal, engine

ARTIFACTS = Path("artifacts")


def slugify(name: str) -> str:
    """Convierte 'Saudi Arabia' → 'saudi-arabia'."""
    return (
        name.lower()
        .replace(" ", "-")
        .replace(".", "")
        .replace("'", "")
        .replace("(", "")
        .replace(")", "")
    )


def seed_teams(db, fixture: pd.DataFrame) -> None:
    """Extrae equipos únicos del fixture con su Elo promedio."""
    home = fixture[["home_team", "home_elo"]].rename(
        columns={"home_team": "team", "home_elo": "elo"}
    )
    away = fixture[["away_team", "away_elo"]].rename(
        columns={"away_team": "team", "away_elo": "elo"}
    )
    team_elos = (
        pd.concat([home, away])
        .dropna(subset=["elo"])
        .groupby("team")["elo"]
        .mean()
        .reset_index()
        .sort_values("elo", ascending=False)
    )

    added = 0
    for _, row in team_elos.iterrows():
        exists = db.query(Team).filter(Team.name == row["team"]).first()
        if not exists:
            db.add(
                Team(
                    name=row["team"],
                    slug=slugify(row["team"]),
                    elo_rating=round(float(row["elo"]), 1),
                )
            )
            added += 1

    db.commit()
    print(f"  Teams: {added} nuevos / {len(team_elos) - added} ya existían")


def seed_matches(db, clean_fixture: pd.DataFrame) -> None:
    """Popula la tabla matches con el fixture original (tiene city y country)."""
    added = 0
    for _, row in clean_fixture.iterrows():
        exists = (
            db.query(Match)
            .filter(
                Match.home_team == row["home_team"],
                Match.away_team == row["away_team"],
                Match.date == row["date"],
            )
            .first()
        )
        if not exists:
            db.add(
                Match(
                    home_team=row["home_team"],
                    away_team=row["away_team"],
                    date=row["date"],
                    city=row.get("city"),
                    country=row.get("country"),
                    # stage y group se pueden completar manualmente después
                )
            )
            added += 1

    db.commit()
    print(f"  Matches: {added} nuevos / {len(clean_fixture) - added} ya existían")


def main() -> None:
    print("Creando tablas si no existen...")
    Base.metadata.create_all(bind=engine)

    # Verificamos que los archivos necesarios existan
    master_path = ARTIFACTS / "master_fixture_2026.csv"
    clean_path  = ARTIFACTS / "clean_fixture_2026.csv"

    missing = [p for p in [master_path, clean_path] if not p.exists()]
    if missing:
        print("ERROR — faltan estos archivos en artifacts/:")
        for p in missing:
            print(f"  {p}")
        print("\nCópialos desde los outputs de las Fases 1-2.")
        sys.exit(1)

    print("Leyendo fixtures...")
    master_fixture = pd.read_csv(master_path, parse_dates=["date"])
    clean_fixture  = pd.read_csv(clean_path,  parse_dates=["date"])

    # Filtramos solo los partidos del Mundial 2026
    clean_wc = clean_fixture[
        clean_fixture["tournament"] == "FIFA World Cup"
    ].copy()

    print(f"  master_fixture_2026: {len(master_fixture)} filas")
    print(f"  clean_fixture_2026 (WC only): {len(clean_wc)} filas")

    with SessionLocal() as db:
        print("\nSeeding teams...")
        seed_teams(db, master_fixture)

        print("Seeding matches...")
        seed_matches(db, clean_wc)

    print("\n✅ Seeding completado.")
    print("   Próximo paso: copia los .pkl a artifacts/ y corre uvicorn.")


if __name__ == "__main__":
    main()