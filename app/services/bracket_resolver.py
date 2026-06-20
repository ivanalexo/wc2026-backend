"""Resolver idempotente del bracket de eliminatorias.

Rellena `home_team`/`away_team` de los partidos de knockout a partir de los
resultados reales guardados en `matches` (status='finished' + scores). No usa
API ni aleatoriedad: las posiciones se calculan con los criterios FIFA
(`tiebreaker.rank_group`) y los terceros con `bracket.assign_third_place_teams`.

Mecanismo de slots (columnas `home_slot`/`away_slot`):
  - "1X" / "2X"  → 1º / 2º del grupo X (cuando el grupo cerró sus 6 partidos)
  - "3rd"        → mejor tercero asignado al slot (cuando cerraron los 12 grupos)
  - "W<n>" / "L<n>" → ganador / perdedor del match número n (cuando n terminó)

Idempotente: solo escribe cuando el slot ya está resuelto y el equipo difiere.
Pensado para llamarse tras cada sync de resultados.

LIMITACIÓN CONOCIDA: un partido de knockout empatado en tiempo reglamentario se
define por penales, pero el modelo de datos aún no guarda penales/ganador. En
ese caso no se propaga el ganador (se loguea). Se resolverá al agregar el campo.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.db.models.match import Match
from app.ml.bracket import assign_third_place_teams
from app.services.tiebreaker import GroupMatch, rank_group

logger = logging.getLogger(__name__)

GROUP_LETTERS = [chr(c) for c in range(ord("A"), ord("L") + 1)]  # A..L
MATCHES_PER_GROUP = 6


@dataclass
class ResolveResult:
    groups_resolved: int = 0
    thirds_assigned: bool = False
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


def _compute_group_positions(matches: list[Match]) -> tuple[dict, dict, list]:
    """Calcula 1º/2º/3º de cada grupo que ya cerró sus 6 partidos."""
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
    thirds: list[dict] = []
    for g, gm in by_group.items():
        if len(gm) < MATCHES_PER_GROUP:
            continue  # grupo incompleto
        teams = sorted({m.home for m in gm} | {m.away for m in gm})
        table = rank_group(teams, gm)
        if len(table) < 3:
            continue
        firsts[g] = table[0]["team"]
        seconds[g] = table[1]["team"]
        third = dict(table[2])
        third["group"] = g
        thirds.append(third)
    return firsts, seconds, thirds


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


def _assign_thirds(matches: list[Match], thirds: list[dict]) -> int:
    """Asigna los 8 mejores terceros a sus slots (solo con los 12 grupos cerrados)."""
    assignment = assign_third_place_teams(thirds)  # {match_number: team}
    by_num = {m.match_number: m for m in matches}
    filled = 0
    for num, team in assignment.items():
        m = by_num.get(num)
        if m is None:
            continue
        side = "home" if m.home_slot == "3rd" else "away"
        filled += _set_team(m, side, team)
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
    logger.warning(
        "Match %s empatado (%s-%s): definición por penales no soportada aún; "
        "no se propaga el ganador.",
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

    firsts, seconds, thirds = _compute_group_positions(matches)
    res = ResolveResult(groups_resolved=len(firsts))
    res.slots_filled += _fill_group_slots(matches, firsts, seconds)

    if len(firsts) == len(GROUP_LETTERS):  # 12 grupos cerrados
        res.slots_filled += _assign_thirds(matches, thirds)
        res.thirds_assigned = True

    res.knockout_propagated = _propagate_knockout(matches)

    db.commit()
    logger.info(
        "Bracket resuelto: grupos=%d/12 thirds=%s slots_grupo=%d knockout=%d",
        res.groups_resolved, res.thirds_assigned, res.slots_filled, res.knockout_propagated,
    )
    return res
