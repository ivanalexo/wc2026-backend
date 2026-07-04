import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import requests
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models.match import Match

logger = logging.getLogger(__name__)

BASE_URL        = "https://api.football-data.org/v4"
WC_CODE         = "WC"           # código de la FIFA World Cup en football-data.org
MIN_REQUESTS_OK = 2              # umbral mínimo de requests disponibles antes de pausar

STATUS_MAP: dict[str, str] = {
    "FINISHED":   "finished",
    "IN_PLAY":    "live",
    "PAUSED":     "live",
    "TIMED":      "scheduled",
    "SCHEDULED":  "scheduled",
    "POSTPONED":  "scheduled",
    "SUSPENDED":  "scheduled",
    "CANCELLED":  "scheduled",
    "AWARDED":    "finished",
}

TEAM_ALIASES: dict[str, str] = {
    "Korea Republic":                "South Korea",
    "Republic of Korea":             "South Korea",
    "USA":                           "United States",
    "United States":                 "United States",
    "Bosnia and Herzegovina":        "Bosnia and Herzegovina",
    "Bosnia-H.":                     "Bosnia and Herzegovina",
    "Bosnia-Herzegovina":            "Bosnia and Herzegovina",
    "DR Congo":                      "DR Congo",
    "Congo DR":                      "DR Congo",
    "Democratic Republic of Congo":  "DR Congo",
    "Cape Verde Islands":            "Cape Verde",
    "Cabo Verde":                    "Cape Verde",
    "Türkiye":                       "Turkey",
    "Curacao":                       "Curaçao",
    "Curaçao":                       "Curaçao",
    "Czechia":                        "Czech Republic",
}


@dataclass
class SyncResult:
    updated: int = 0
    skipped: int = 0
    not_found: int = 0
    error: str | None = None
    requests_available: int | None = None

def _resolve(name: str) -> str:
    return TEAM_ALIASES.get(name.strip(), name.strip())


def _find_match(db: Session, home: str, away: str) -> tuple[Match | None, bool]:
    m = db.query(Match).filter(Match.home_team == home, Match.away_team == away).first()
    if m:
        return m, False
    m = db.query(Match).filter(Match.home_team == away, Match.away_team == home).first()
    if m:
        return m, True

    m = db.query(Match).filter(Match.away_slot == "3rd", Match.home_team == home).first()
    if m:
        if m.away_team != away:
            logger.info("Adoptando tercero real del API en match %s: %s", m.match_number, away)
            m.away_team = away
        return m, False
    m = db.query(Match).filter(Match.away_slot == "3rd", Match.home_team == away).first()
    if m:
        if m.away_team != home:
            logger.info("Adoptando tercero real del API en match %s: %s", m.match_number, home)
            m.away_team = home
        return m, True

    return None, False


def _check_rate_limit(headers: dict) -> None:
    """Pausa si quedan muy pocos requests disponibles."""
    try:
        available = int(headers.get("X-RequestsAvailable", 99))
        reset_secs = int(headers.get("X-RequestCounter-Reset", 0))
        logger.info("Rate limit — disponibles: %d | reset en: %ds", available, reset_secs)
        if available < MIN_REQUESTS_OK:
            wait = max(reset_secs, 5)
            logger.warning("Pocos requests disponibles (%d). Esperando %ds...", available, wait)
            time.sleep(wait)
    except (ValueError, TypeError):
        pass


def sync_wc_results(db: Session) -> SyncResult:
    """
    Llama a football-data.org y actualiza status/scores en la DB.
    Retorna un SyncResult con el resumen.
    """
    key = settings.football_data_org_key
    if not key:
        return SyncResult(error="FOOTBALL_DATA_ORG_KEY no configurada")

    headers = {"X-Auth-Token": key}

    url = f"{BASE_URL}/competitions/{WC_CODE}/matches"
    params = {"status": "FINISHED,IN_PLAY,PAUSED"}

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=15)
    except requests.RequestException as exc:
        return SyncResult(error=f"Error de red: {exc}")

    _check_rate_limit(dict(resp.headers))
    requests_available = None
    try:
        requests_available = int(resp.headers.get("X-RequestsAvailable", -1))
    except (ValueError, TypeError):
        pass

    if resp.status_code == 429:
        reset = resp.headers.get("X-RequestCounter-Reset", "desconocido")
        return SyncResult(error=f"Rate limit alcanzado. Reset en {reset}s", requests_available=0)

    if resp.status_code == 403:
        return SyncResult(error="API key inválida o sin acceso a esta competición")

    if not resp.ok:
        return SyncResult(error=f"HTTP {resp.status_code}: {resp.text[:200]}")

    data = resp.json()
    matches_api = data.get("matches", [])
    logger.info("football-data.org devolvió %d partidos", len(matches_api))

    result = SyncResult(requests_available=requests_available)

    for m_api in matches_api:
        api_home  = _resolve(m_api["homeTeam"]["name"])
        api_away  = _resolve(m_api["awayTeam"]["name"])
        api_status = STATUS_MAP.get(m_api.get("status", ""), "scheduled")

        score_full = m_api.get("score", {}).get("fullTime", {})
        api_home_score = score_full.get("home")
        api_away_score = score_full.get("away")
        api_winner = m_api.get("score", {}).get("winner")

        match, is_reversed = _find_match(db, api_home, api_away)

        if match is None:
            logger.warning("Partido no encontrado en DB: %s vs %s", api_home, api_away)
            result.not_found += 1
            continue

        changed = False

        # Asignar scores respetando si el orden está invertido en la DB
        if api_home_score is not None and api_away_score is not None:
            db_home_score = api_away_score if is_reversed else api_home_score
            db_away_score = api_home_score if is_reversed else api_away_score
            if match.home_score != db_home_score or match.away_score != db_away_score:
                match.home_score = db_home_score
                match.away_score = db_away_score
                changed = True

        if (
            api_home_score is not None
            and api_away_score is not None
            and api_home_score == api_away_score
        ):
            if api_winner == "HOME_TEAM":
                new_winner = "AWAY" if is_reversed else "HOME"
            elif api_winner == "AWAY_TEAM":
                new_winner = "HOME" if is_reversed else "AWAY"
            else:
                new_winner = None
            if new_winner and match.winner != new_winner:
                match.winner = new_winner
                changed = True

        if match.status != api_status:
            match.status = api_status
            changed = True

        if changed:
            result.updated += 1
            logger.info(
                "Actualizado: %s %s vs %s %s — status=%s",
                match.home_score, match.home_team,
                match.away_team, match.away_score,
                api_status,
            )
        else:
            result.skipped += 1

    db.commit()

    try:
        from app.services.bracket_resolver import resolve_bracket
        bracket = resolve_bracket(db)
        logger.info(
            "Resolver bracket: grupos=%d/12 slots=%d knockout=%d",
            bracket.groups_resolved, bracket.slots_filled, bracket.knockout_propagated,
        )
    except Exception:
        logger.exception("Error resolviendo el bracket tras el sync")

    return result
