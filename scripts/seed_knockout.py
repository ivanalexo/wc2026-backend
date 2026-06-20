# =============================================================================
# Siembra la estructura del bracket de eliminatorias (Round of 32 → Final).
#   python scripts/seed_knockout.py
# =============================================================================

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db.models.match import Match
from app.db.session import SessionLocal

# Tupla: (match_number, fecha UTC "YYYY-MM-DD HH:MM", city, country, stage, home_slot, away_slot)
KNOCKOUT: list[tuple[int, str, str, str, str, str, str]] = [
    # --- ronda 32 ---
    (73, "2026-06-28 19:00", "Inglewood",       "United States", "Round of 32", "2A", "2B"),
    (74, "2026-06-29 20:30", "Foxborough",      "United States", "Round of 32", "1E", "3rd"),
    (75, "2026-06-30 01:00", "Guadalupe",       "Mexico",        "Round of 32", "1F", "2C"),
    (76, "2026-06-29 17:00", "Houston",         "United States", "Round of 32", "1C", "2F"),
    (77, "2026-06-30 21:00", "East Rutherford", "United States", "Round of 32", "1I", "3rd"),
    (78, "2026-06-30 17:00", "Arlington",       "United States", "Round of 32", "2E", "2I"),
    (79, "2026-07-01 01:00", "Mexico City",     "Mexico",        "Round of 32", "1A", "3rd"),
    (80, "2026-07-01 16:00", "Atlanta",         "United States", "Round of 32", "1L", "3rd"),
    (81, "2026-07-02 00:00", "Santa Clara",     "United States", "Round of 32", "1D", "3rd"),
    (82, "2026-07-01 20:00", "Seattle",         "United States", "Round of 32", "1G", "3rd"),
    (83, "2026-07-02 23:00", "Toronto",         "Canada",        "Round of 32", "2K", "2L"),
    (84, "2026-07-02 19:00", "Inglewood",       "United States", "Round of 32", "1H", "2J"),
    (85, "2026-07-03 03:00", "Vancouver",       "Canada",        "Round of 32", "1B", "3rd"),
    (86, "2026-07-03 22:00", "Miami",           "United States", "Round of 32", "1J", "2H"),
    (87, "2026-07-04 01:30", "Kansas City",     "United States", "Round of 32", "1K", "3rd"),
    (88, "2026-07-03 18:00", "Arlington",       "United States", "Round of 32", "2D", "2G"),
    # --- ocatvos ---
    (89, "2026-07-04 21:00", "Philadelphia",    "United States", "Round of 16", "W74", "W77"),
    (90, "2026-07-04 17:00", "Houston",         "United States", "Round of 16", "W73", "W75"),
    (91, "2026-07-05 20:00", "East Rutherford", "United States", "Round of 16", "W76", "W78"),
    (92, "2026-07-06 00:00", "Mexico City",     "Mexico",        "Round of 16", "W79", "W80"),
    (93, "2026-07-06 19:00", "Arlington",       "United States", "Round of 16", "W83", "W84"),
    (94, "2026-07-07 00:00", "Seattle",         "United States", "Round of 16", "W81", "W82"),
    (95, "2026-07-07 16:00", "Atlanta",         "United States", "Round of 16", "W86", "W88"),
    (96, "2026-07-07 20:00", "Vancouver",       "Canada",        "Round of 16", "W85", "W87"),
    # --- cuartos ---
    (97,  "2026-07-09 20:00", "Foxborough",     "United States", "Quarter-finals", "W89", "W90"),
    (98,  "2026-07-10 19:00", "Inglewood",      "United States", "Quarter-finals", "W93", "W94"),
    (99,  "2026-07-11 21:00", "Miami",          "United States", "Quarter-finals", "W91", "W92"),
    (100, "2026-07-12 01:00", "Kansas City",    "United States", "Quarter-finals", "W95", "W96"),
    # --- semis ---
    (101, "2026-07-14 19:00", "Arlington",      "United States", "Semi-finals", "W97", "W98"),
    (102, "2026-07-15 19:00", "Atlanta",        "United States", "Semi-finals", "W99", "W100"),
    # --- tercer y final ---
    (103, "2026-07-18 21:00", "Miami",          "United States", "3rd Place", "L101", "L102"),
    (104, "2026-07-19 19:00", "East Rutherford","United States", "Final",     "W101", "W102"),
]


def parse_utc(s: str) -> datetime:
    """'YYYY-MM-DD HH:MM' UTC → datetime UTC naive (mismo patrón que seed_schedule.py)."""
    return datetime.strptime(s, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc).replace(tzinfo=None)


def backfill_group_numbers(db) -> None:
    """Asigna match_number 1–72 a los partidos de grupos en orden cronológico.
    """
    groups = (
        db.query(Match)
        .filter(Match.stage == "Group Stage")
        .order_by(Match.date, Match.home_team, Match.away_team)
        .all()
    )
    if len(groups) != 72:
        print(f"Se esperaban 72 partidos de grupos, hay {len(groups)}. "
              f"¿Corriste seed.py?.")

    changed = 0
    for i, m in enumerate(groups, start=1):
        if m.match_number != i:
            m.match_number = i
            changed += 1
    print(f"  Grupos  → {len(groups)} numerados (1–{len(groups)}) | {changed} actualizados")


def seed_knockout_matches(db) -> None:
    added = updated = skipped = 0

    for num, dt_str, city, country, stage, home_slot, away_slot in KNOCKOUT:
        date = parse_utc(dt_str)
        existing = db.query(Match).filter(Match.match_number == num).first()

        if existing:
            fields = {
                "date": date, "city": city, "country": country, "stage": stage,
                "home_slot": home_slot, "away_slot": away_slot,
            }
            if any(getattr(existing, k) != v for k, v in fields.items()):
                for k, v in fields.items():
                    setattr(existing, k, v)
                updated += 1
            else:
                skipped += 1
        else:
            db.add(Match(
                match_number=num,
                home_team=None,
                away_team=None,
                home_slot=home_slot,
                away_slot=away_slot,
                date=date,
                city=city,
                country=country,
                stage=stage,
                status="scheduled",
            ))
            added += 1

    print(f"  Knockout → {added} nuevos | {updated} actualizados | {skipped} sin cambios")


def main() -> None:
    print("Sembrando bracket de eliminatorias...")
    with SessionLocal() as db:
        backfill_group_numbers(db)
        seed_knockout_matches(db)
        db.commit()
    print("\n[OK] Bracket sembrado. Equipos en NULL hasta que el resolver los rellene.")


if __name__ == "__main__":
    main()
