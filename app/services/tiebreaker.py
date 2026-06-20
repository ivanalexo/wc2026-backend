"""Desempate de grupos según criterios FIFA (WC2026).

Orden oficial:
  1. Puntos (todos los partidos del grupo)
  2. Diferencia de goles global
  3. Goles a favor global
  4. Head-to-head entre empatados (pts → dif → GF SOLO en partidos entre ellos),
     reaplicado recursivamente a los subconjuntos que sigan empatados.
  5. Fair play / 6. Sorteo  → NO calculables desde marcadores: fallback determinista
     (orden alfabético) + log de advertencia.

Todo se calcula desde los goles de los partidos; no requiere datos extra.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GroupMatch:
    home: str
    away: str
    home_score: int
    away_score: int


def _accumulate(teams: list[str], matches: list[GroupMatch]) -> dict[str, dict]:
    """pts/gf/ga/gd por equipo, contando solo partidos entre los `teams` dados."""
    tset = set(teams)
    stats = {t: {"pts": 0, "gf": 0, "ga": 0} for t in teams}
    for m in matches:
        if m.home not in tset or m.away not in tset:
            continue
        stats[m.home]["gf"] += m.home_score
        stats[m.home]["ga"] += m.away_score
        stats[m.away]["gf"] += m.away_score
        stats[m.away]["ga"] += m.home_score
        if m.home_score > m.away_score:
            stats[m.home]["pts"] += 3
        elif m.away_score > m.home_score:
            stats[m.away]["pts"] += 3
        else:
            stats[m.home]["pts"] += 1
            stats[m.away]["pts"] += 1
    for t in stats:
        stats[t]["gd"] = stats[t]["gf"] - stats[t]["ga"]
    return stats


def _key(stats: dict, t: str) -> tuple[int, int, int]:
    s = stats[t]
    return (s["pts"], s["gd"], s["gf"])


def _clusters(ordered: list[str], keyfn) -> list[list[str]]:
    """Agrupa equipos consecutivos con la misma clave (ya vienen ordenados)."""
    out: list[list[str]] = []
    cur: list[str] = []
    last = None
    for t in ordered:
        k = keyfn(t)
        if cur and k != last:
            out.append(cur)
            cur = []
        cur.append(t)
        last = k
    if cur:
        out.append(cur)
    return out


def _fallback(teams: list[str]) -> list[str]:
    logger.warning(
        "Empate no resoluble por marcadores (requiere fair play/sorteo FIFA): %s. "
        "Se aplica orden alfabético determinista.", sorted(teams),
    )
    return sorted(teams)


def _break_h2h(cluster: list[str], all_matches: list[GroupMatch]) -> list[str]:
    """Resuelve un grupo empatado por head-to-head, recursivo para subconjuntos."""
    h2h = _accumulate(cluster, all_matches)
    ordered = sorted(cluster, key=lambda t: _key(h2h, t), reverse=True)
    out: list[str] = []
    for sub in _clusters(ordered, lambda t: _key(h2h, t)):
        if len(sub) == 1:
            out.append(sub[0])
        elif len(sub) < len(cluster):
            # Subconjunto menor → reaplicar head-to-head solo entre ellos (FIFA).
            out.extend(_break_h2h(sub, all_matches))
        else:
            out.extend(_fallback(sub))
    return out


def rank_group(teams: list[str], matches: list[GroupMatch]) -> list[dict]:
    """Ordena los equipos del grupo (1º..Nº) y devuelve sus stats globales.

    Cada elemento: {"team", "pts", "gf", "ga", "gd"}.
    """
    overall = _accumulate(teams, matches)
    ordered = sorted(teams, key=lambda t: _key(overall, t), reverse=True)
    final: list[str] = []
    for cluster in _clusters(ordered, lambda t: _key(overall, t)):
        if len(cluster) == 1:
            final.append(cluster[0])
        else:
            final.extend(_break_h2h(cluster, matches))
    return [{"team": t, **overall[t]} for t in final]
