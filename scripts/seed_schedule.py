# =============================================================================
# Actualiza horarios y resultados de partidos desde el CSV oficial del torneo.
#
# Ejecutar desde la raíz del proyecto (wc2026-backend/):
#   python scripts/seed_schedule.py
#   python scripts/seed_schedule.py --csv ../world-cup_2026.csv
# =============================================================================

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db.models.match import Match
from app.db.session import SessionLocal

CSV_ALIASES: dict[str, str] = {
    "Bosnia & Herzegovina":          "Bosnia and Herzegovina",
    "Bosnia-Herzegovina":            "Bosnia and Herzegovina",
    "Congo DR":                      "DR Congo",
    "Congo, DR":                     "DR Congo",
    "Democratic Republic of Congo":  "DR Congo",
    "Cape Verde Islands":            "Cape Verde",
    "Cabo Verde":                    "Cape Verde",
    "Türkiye":                       "Turkey",
    "USA":                           "United States",
    "Korea Republic":                "South Korea",
    "Republic of Korea":             "South Korea",
    "Czechia":                       "Czech Republic",
}

STAGE_KEYWORDS: list[tuple[str, str]] = [
    ("Round of 32",    "Round of 32"),
    ("Round of 16",    "Round of 16"),
    ("Quarter-finals", "Quarter-finals"),
    ("Semi-finals",    "Semi-finals"),
    ("3rd Place",      "3rd Place"),
    ("Final",          "Final"),
]


def resolve(name: str) -> str:
    return CSV_ALIASES.get(name.strip(), name.strip())


def parse_datetime_utc(date_str: str, time_str: str) -> datetime:
    """Combina date YYYY-MM-DD + time HH:MM en un datetime UTC naive (para PostgreSQL)."""
    dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
    return dt.replace(tzinfo=timezone.utc).replace(tzinfo=None)  # naive UTC para la DB


def parse_result(result: str) -> tuple[int, int] | tuple[None, None]:
    """Parsea '2-0' → (2, 0). Retorna (None, None) si está vacío o inválido."""
    if not result or pd.isna(result):
        return None, None
    parts = str(result).strip().split("-")
    if len(parts) != 2:
        return None, None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None, None


def infer_stage_from_tbd(team_str: str) -> str | None:
    for keyword, stage in STAGE_KEYWORDS:
        if keyword in team_str:
            return stage
    return None


def find_match(db, home: str, away: str) -> tuple["Match | None", bool]:
    """
    Busca el partido en DB. Retorna (match, reversed).
    reversed=True significa que en la DB el orden home/away está invertido vs el CSV.
    """
    match = db.query(Match).filter(Match.home_team == home, Match.away_team == away).first()
    if match:
        return match, False
    match = db.query(Match).filter(Match.home_team == away, Match.away_team == home).first()
    if match:
        return match, True
    return None, False


def main(csv_path: Path) -> None:
    if not csv_path.exists():
        print(f"ERROR — No se encontró el CSV: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path, dtype=str)
    print(f"CSV cargado: {len(df)} filas desde {csv_path}")

    updated_date = 0
    updated_score = 0
    updated_status = 0
    skipped_tbd = 0
    not_found = []

    with SessionLocal() as db:
        for _, row in df.iterrows():
            raw_home = row.get("home_team", "")
            raw_away = row.get("away_team", "")

            # Filas de knockout con equipos TBD — omitir por ahora
            if str(raw_home).startswith("TBD"):
                skipped_tbd += 1
                continue

            home = resolve(raw_home)
            away = resolve(raw_away)

            new_date = parse_datetime_utc(row["date"], row["time"])
            home_score, away_score = parse_result(row.get("result", ""))

            csv_status_raw = str(row.get("status", "")).strip().lower()
            new_status = "finished" if csv_status_raw == "played" else "scheduled"

            match, is_reversed = find_match(db, home, away)

            if match is None:
                not_found.append(f"{home} vs {away}  ({row['date']} {row['time']})")
                continue

            changed = False

            # Actualizar fecha/hora si difiere (tolerancia: ignoramos segundos)
            if match.date is None or match.date.replace(second=0) != new_date.replace(second=0):
                match.date = new_date
                changed = True
                updated_date += 1

            # Actualizar ciudad si el CSV la trae y la DB no la tiene o difiere
            csv_city = str(row.get("city", "")).strip() or None
            if csv_city and match.city != csv_city:
                match.city = csv_city
                changed = True

            # Actualizar resultado — si el orden está invertido, invertir también los scores
            if home_score is not None:
                db_home_score = away_score if is_reversed else home_score
                db_away_score = home_score if is_reversed else away_score
                if match.home_score != db_home_score or match.away_score != db_away_score:
                    match.home_score = db_home_score
                    match.away_score = db_away_score
                    changed = True
                    updated_score += 1

            # Actualizar status
            if match.status != new_status:
                match.status = new_status
                changed = True
                updated_status += 1

        db.commit()

    print(f"\nResultados:")
    print(f"  Fechas/horas actualizadas : {updated_date}")
    print(f"  Resultados actualizados   : {updated_score}")
    print(f"  Status actualizados       : {updated_status}")
    print(f"  Filas TBD omitidas        : {skipped_tbd}")

    if not_found:
        print(f"\n  [!] {len(not_found)} partido(s) no encontrados en la DB:")
        for m in not_found:
            print(f"     {m}")
        print("  -> Verifica los nombres en CSV_ALIASES o ejecuta seed.py primero.")
    else:
        print("\n[OK] Todos los partidos del grupo encontrados y actualizados.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Actualiza horarios y resultados desde CSV")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "world-cup_2026.csv",
        help="Ruta al CSV del torneo (default: ../world-cup_2026.csv relativo al proyecto)",
    )
    args = parser.parse_args()
    main(args.csv)
