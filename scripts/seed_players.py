#!/usr/bin/env python3
"""
Seed WC2026 squad players from API Football (https://v3.football.api-sports.io).

Usage:
    python scripts/seed_players.py
    python scripts/seed_players.py --start-from "Team Name"
    python scripts/seed_players.py --dry-run

Requires API_FOOTBALL_KEY in .env
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from app.db.models import Player
from app.db.session import SessionLocal

API_BASE = "https://v3.football.api-sports.io"
CSV_PATH = Path("artifacts") / "master_fixture_2026.csv"

TEAM_NAME_MAP: dict[str, str] = {
    "Ivory Coast": "Cote d'Ivoire",
    "Ivory Coast": "Ivory Coast",
    "DR Congo": "Congo DR",
    "United States": "USA",
    "South Korea": "Korea Republic",
    "Czech Republic": "Czechia",
    "Bosnia and Herzegovina": "Bosnia",
    "Curaçao": "Curacao",
}

REQUEST_DELAY = 6.5  # seconds


def get_teams_from_csv() -> list[str]:
    teams: set[str] = set()
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            teams.add(row["home_team"])
            teams.add(row["away_team"])
    return sorted(teams)


def find_team_id(client: httpx.Client, csv_name: str) -> int | None:
    api_name = TEAM_NAME_MAP.get(csv_name, csv_name)

    resp = client.get(f"{API_BASE}/teams", params={"name": api_name, "type": "National"})
    resp.raise_for_status()
    results = resp.json().get("response", [])
    if results:
        return results[0]["team"]["id"]

    time.sleep(REQUEST_DELAY)

    resp = client.get(f"{API_BASE}/teams", params={"search": api_name})
    resp.raise_for_status()
    for item in resp.json().get("response", []):
        if item["team"].get("national") or item["team"].get("type") == "National":
            return item["team"]["id"]

    return None


def fetch_squad(client: httpx.Client, team_id: int) -> list[dict]:
    resp = client.get(f"{API_BASE}/players/squads", params={"team": team_id})
    resp.raise_for_status()
    data = resp.json().get("response", [])
    if not data:
        return []
    return data[0].get("players", [])


def upsert_players(db, team_name: str, players: list[dict], dry_run: bool) -> int:
    count = 0
    for p in players:
        api_id: int | None = p.get("id")
        name: str = p.get("name") or ""
        if not name:
            continue

        if not dry_run:
            existing = (
                db.query(Player).filter(Player.api_id == api_id).first()
                if api_id
                else None
            )
            if existing:
                existing.team_name = team_name
                existing.name = name
                existing.position = p.get("position")
                existing.number = p.get("number")
                existing.age = p.get("age")
                existing.photo_url = p.get("photo")
            else:
                db.add(
                    Player(
                        team_name=team_name,
                        api_id=api_id,
                        name=name,
                        position=p.get("position"),
                        number=p.get("number"),
                        age=p.get("age"),
                        photo_url=p.get("photo"),
                    )
                )
        count += 1

    if not dry_run:
        db.commit()

    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed WC2026 players from API Football")
    parser.add_argument("--dry-run", action="store_true", help="Fetch data but do not write to DB")
    parser.add_argument("--start-from", metavar="TEAM", help="Resume from this team name (inclusive)")
    args = parser.parse_args()

    api_key = settings.api_football_key
    if not api_key:
        print("ERROR: API_FOOTBALL_KEY is not set in .env")
        sys.exit(1)

    teams = get_teams_from_csv()

    if args.start_from:
        try:
            start_idx = teams.index(args.start_from)
            teams = teams[start_idx:]
            print(f"Resuming from '{args.start_from}' ({len(teams)} teams remaining)")
        except ValueError:
            print(f"ERROR: '{args.start_from}' not found. Available teams:")
            print("\n".join(f"  {t}" for t in teams))
            sys.exit(1)
    else:
        print(f"Teams to seed: {len(teams)}")

    if args.dry_run:
        print("DRY RUN – no DB writes\n")

    headers = {"x-apisports-key": api_key}
    db = SessionLocal()

    try:
        with httpx.Client(headers=headers, timeout=30) as client:
            for i, team_name in enumerate(teams, 1):
                print(f"[{i}/{len(teams)}] {team_name}", end=" ... ", flush=True)

                team_id = find_team_id(client, team_name)
                time.sleep(REQUEST_DELAY)

                if team_id is None:
                    print("team not found in API, skipping")
                    continue

                squad = fetch_squad(client, team_id)
                time.sleep(REQUEST_DELAY)

                if not squad:
                    print(f"no squad data (team_id={team_id}), skipping")
                    continue

                count = upsert_players(db, team_name, squad, dry_run=args.dry_run)
                print(f"{count} players upserted (team_id={team_id})")

    finally:
        db.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
