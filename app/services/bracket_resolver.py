from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.db.models.match import Match
from app.services.tiebreaker import GroupMatch, rank_group

logger = logging.getLogger(__name__)

MATCHES_PER_GROUP = 6


@dataclass
class ResolveResult:
    groups_resolved: int = 0
    slots_filled: int = 0
    knockout_propagated: int = 0


def _set_team(match: Match, side: str, team: str | None) -> int:
    """Asigna el equipo a un lado si está definido y cambió. Retorna 1 si escribió."""
    if team is None:
        return 0
    attr = f"{side}_team"
    if getattr(match, attr) != team:
        setattr(match, attr, team)
        return 1
    return 0


def _compute_group_positions(matches: list[Match]) -> tuple[dict, dict]:
    by_group: dict[str, list[GroupMatch]] = {}
    for m in matches:
        if (
            m.stage == "Group Stage"
            and m.status == "finished"
            and m.home_score is not None
            and m.away_score is not None
            and m.home_team
            and m.away_team
            and m.group
        ):
            by_group.setdefault(m.group, []).append(
                GroupMatch(m.home_team, m.away_team, m.home_score, m.away_score)
            )

    firsts: dict[str, str] = {}
    seconds: dict[str, str] = {}
    for g, gm in by_group.items():
        if len(gm) < MATCHES_PER_GROUP:
            continue  # grupo incompleto
        teams = sorted({m.home for m in gm} | {m.away for m in gm})
        table = rank_group(teams, gm)
        if len(table) < 2:
            continue
        firsts[g] = table[0]["team"]
        seconds[g] = table[1]["team"]
    return firsts, seconds


def _resolve_pos_slot(slot: str | None, firsts: dict, seconds: dict) -> str | None:
    """'1X'/'2X' → nombre del equipo si el grupo está resuelto; si no, None."""
    if not slot or len(slot) != 2 or slot[1] not in "ABCDEFGHIJKL":
        return None
    if slot[0] == "1":
        return firsts.get(slot[1])
    if slot[0] == "2":
        return seconds.get(slot[1])
    return None


def _fill_group_slots(matches: list[Match], firsts: dict, seconds: dict) -> int:
    filled = 0
    for m in matches:
        filled += _set_team(m, "home", _resolve_pos_slot(m.home_slot, firsts, seconds))
        filled += _set_team(m, "away", _resolve_pos_slot(m.away_slot, firsts, seconds))
    return filled


def _winner_loser(match: Match) -> tuple[str | None, str | None]:
    """Ganador y perdedor de un partido de knockout terminado. (None, None) si no aplica."""
    if (
        match.status != "finished"
        or match.home_score is None
        or match.away_score is None
        or not match.home_team
        or not match.away_team
    ):
        return None, None
    if match.home_score > match.away_score:
        return match.home_team, match.away_team
    if match.away_score > match.home_score:
        return match.away_team, match.home_team
    if match.winner == "HOME":
        return match.home_team, match.away_team
    if match.winner == "AWAY":
        return match.away_team, match.home_team
    logger.warning(
        "Match %s empatado (%s-%s) sin 'winner' registrado; no se propaga el ganador.",
        match.match_number, match.home_score, match.away_score,
    )
    return None, None


def _propagate_knockout(matches: list[Match]) -> int:
    """Propaga ganadores/perdedores a los slots W<n>/L<n> hasta punto fijo."""
    by_num = {m.match_number: m for m in matches if m.match_number}
    total = 0
    while True:
        winners: dict[int, str] = {}
        losers: dict[int, str] = {}
        for num, m in by_num.items():
            w, loser = _winner_loser(m)
            if w:
                winners[num] = w
                losers[num] = loser

        filled = 0
        for m in matches:
            for side in ("home", "away"):
                slot = getattr(m, f"{side}_slot")
                if not slot or slot[0] not in ("W", "L") or not slot[1:].isdigit():
                    continue
                n = int(slot[1:])
                team = winners.get(n) if slot[0] == "W" else losers.get(n)
                filled += _set_team(m, side, team)
        total += filled
        if filled == 0:  # punto fijo: nada nuevo que propagar
            break
    return total


def resolve_bracket(db: Session) -> ResolveResult:
    """Resuelve todo lo resoluble del bracket con el estado actual de la DB."""
    matches = db.query(Match).all()

    firsts, seconds = _compute_group_positions(matches)
    res = ResolveResult(groups_resolved=len(firsts))
    res.slots_filled += _fill_group_slots(matches, firsts, seconds)
    res.knockout_propagated = _propagate_knockout(matches)

    db.commit()
    logger.info(
        "Bracket resuelto: grupos=%d/12 slots_grupo=%d knockout=%d",
        res.groups_resolved, res.slots_filled, res.knockout_propagated,
    )
    return res
