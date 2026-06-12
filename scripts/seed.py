# =============================================================================
# Ejecutar desde la raíz del proyecto (wc2026-backend/):
#   python scripts/seed.py
# =============================================================================

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db.base import Base
from app.db.models.match import Match
from app.db.models.team import Team
from app.db.session import SessionLocal, engine

ARTIFACTS = Path("artifacts")

GROUP_MAP: dict[str, str] = {
    # Group A
    "Mexico":       "A",
    "South Korea":  "A",
    "South Africa": "A",
    "Czech Republic": "A",
    # Group B
    "Canada":               "B",
    "Switzerland":          "B",
    "Qatar":                "B",
    "Bosnia and Herzegovina": "B",
    # Group C
    "Brazil":   "C",
    "Morocco":  "C",
    "Scotland": "C",
    "Haiti":    "C",
    # Group D
    "United States": "D",
    "Paraguay":      "D",
    "Australia":     "D",
    "Turkey":        "D",
    # Group E
    "Germany":     "E",
    "Curacao":     "E",
    "Ivory Coast": "E",
    "Ecuador":     "E",
    # Group F
    "Netherlands": "F",
    "Japan":       "F",
    "Tunisia":     "F",
    "Sweden":      "F",
    # Group G
    "Belgium":     "G",
    "Egypt":       "G",
    "Iran":        "G",
    "New Zealand": "G",
    # Group H
    "Spain":        "H",
    "Cape Verde":   "H",
    "Saudi Arabia": "H",
    "Uruguay":      "H",
    # Group I
    "France":  "I",
    "Senegal": "I",
    "Norway":  "I",
    "Iraq":    "I",
    # Group J
    "Argentina": "J",
    "Algeria":   "J",
    "Austria":   "J",
    "Jordan":    "J",
    # Group K
    "Portugal":   "K",
    "Colombia":   "K",
    "Uzbekistan": "K",
    "DR Congo":   "K",
    # Group L
    "England": "L",
    "Croatia": "L",
    "Ghana":   "L",
    "Panama":  "L",
}

# Aliases para nombres alternativos que pueden aparecer en el dataset
TEAM_ALIASES: dict[str, str] = {
    "USA":                    "United States",
    "Czechia":                "Czech Republic",
    "Bosnia-Herzegovina":     "Bosnia and Herzegovina",
    "Curaçao":                "Curacao",
    "Côte d'Ivoire":          "Ivory Coast",
    "Cabo Verde":             "Cape Verde",
    "Türkiye":                "Turkey",
    "Korea Republic":         "South Korea",
    "Republic of Korea":      "South Korea",
    "Congo DR":               "DR Congo",
    "Democratic Republic of Congo": "DR Congo",
}


def resolve_team(name: str) -> str:
    """Devuelve el nombre canónico del equipo resolviendo aliases."""
    return TEAM_ALIASES.get(name, name)


def get_group(team_name: str) -> str | None:
    canonical = resolve_team(team_name)
    return GROUP_MAP.get(canonical)


def slugify(name: str) -> str:
    return (
        name.lower()
        .replace(" ", "-")
        .replace(".", "")
        .replace("'", "")
        .replace("(", "")
        .replace(")", "")
        .replace("ç", "c")
        .replace("ü", "u")
        .replace("é", "e")
        .replace("ô", "o")
    )

def seed_teams(db, fixture: pd.DataFrame) -> None:
    """
    Extrae los 48 equipos del fixture, asigna Elo promedio y grupo.
    """
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

    added = updated = skipped = 0
    for _, row in team_elos.iterrows():
        name  = row["team"]
        group = get_group(name)

        existing = db.query(Team).filter(Team.name == name).first()
        if existing:
            # Actualizamos el grupo si no lo tenía
            if existing.group is None and group:
                existing.group = group
                updated += 1
            else:
                skipped += 1
        else:
            db.add(Team(
                name=name,
                slug=slugify(name),
                elo_rating=round(float(row["elo"]), 1),
                group=group,
            ))
            added += 1

    db.commit()
    print(f"  Teams  → {added} nuevos | {updated} actualizados | {skipped} sin cambios")


def seed_matches(db, clean_fixture: pd.DataFrame) -> None:
    """
    Popula la tabla matches. Todos los 72 partidos son 'Group Stage'.
    El grupo se deriva del GROUP_MAP usando home_team.
    """
    added = updated = skipped = 0
    no_group = []

    for _, row in clean_fixture.iterrows():
        home  = row["home_team"]
        away  = row["away_team"]
        group = get_group(home) or get_group(away)

        if group is None:
            no_group.append(f"{home} vs {away}")

        existing = (
            db.query(Match)
            .filter(
                Match.home_team == home,
                Match.away_team == away,
                Match.date == row["date"],
            )
            .first()
        )

        if existing:
            if existing.stage is None or existing.group is None:
                existing.stage = "Group Stage"
                existing.group = group
                updated += 1
            else:
                skipped += 1
        else:
            db.add(Match(
                home_team=home,
                away_team=away,
                date=row["date"],
                city=row.get("city"),
                country=row.get("country"),
                stage="Group Stage",
                group=group,
            ))
            added += 1

    db.commit()
    print(f"  Matches → {added} nuevos | {updated} actualizados | {skipped} sin cambios")

    if no_group:
        print(f"\n  ⚠️  {len(no_group)} partido(s) sin grupo asignado:")
        for m in no_group:
            print(f"     {m}")
        print("  → Verifica los nombres en GROUP_MAP o agrega un alias en TEAM_ALIASES.")

def main() -> None:
    print("Verificando tablas...")
    Base.metadata.create_all(bind=engine)

    master_path = ARTIFACTS / "master_fixture_2026.csv"
    clean_path  = ARTIFACTS / "clean_fixture_2026.csv"

    missing = [p for p in [master_path, clean_path] if not p.exists()]
    if missing:
        print("ERROR — faltan archivos en artifacts/:")
        for p in missing:
            print(f"  {p}")
        sys.exit(1)

    print("Leyendo fixtures...")
    master_fixture = pd.read_csv(master_path, parse_dates=["date"])
    clean_fixture  = pd.read_csv(clean_path,  parse_dates=["date"])

    clean_wc = clean_fixture[
        clean_fixture["tournament"] == "FIFA World Cup"
    ].copy()

    print(f"  master_fixture_2026 : {len(master_fixture)} partidos")
    print(f"  clean_fixture (WC)  : {len(clean_wc)} partidos")
    print(f"  Todos son 'Group Stage' (June 11-27)")

    with SessionLocal() as db:
        print("\nSeeding teams...")
        seed_teams(db, master_fixture)

        print("Seeding matches...")
        seed_matches(db, clean_wc)

    print("\n✅ Seeding completado.")
    print("   Teams y matches tienen stage='Group Stage' y group asignado.")
    print("   Los partidos de Round of 32 en adelante se agregarán durante el torneo.")


if __name__ == "__main__":
    main()