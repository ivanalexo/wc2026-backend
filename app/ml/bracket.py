"""Estructura del bracket de eliminatorias WC2026 (fuente única).
"""
from __future__ import annotations

R32_SLOTS = [
    (73, "2A",  "2B",  None),
    (74, "1E",  "3rd", "ABCDF"),
    (75, "1F",  "2C",  None),
    (76, "1C",  "2F",  None),
    (77, "1I",  "3rd", "CDFGH"),
    (78, "2E",  "2I",  None),
    (79, "1A",  "3rd", "CEFHI"),
    (80, "1L",  "3rd", "EHIJK"),
    (81, "1D",  "3rd", "BEFIJ"),
    (82, "1G",  "3rd", "AEHIJ"),
    (83, "2K",  "2L",  None),
    (84, "1H",  "2J",  None),
    (85, "1B",  "3rd", "EFGIJ"),
    (86, "1J",  "2H",  None),
    (87, "1K",  "3rd", "DEIJL"),
    (88, "2D",  "2G",  None),
]

R16_PAIRS = [(1, 4), (0, 2), (3, 5), (6, 7), (10, 11), (8, 9), (13, 15), (12, 14)]
QF_PAIRS = [(0, 1), (4, 5), (2, 3), (6, 7)]
SF_PAIRS = [(0, 1), (2, 3)]

THIRD_PLACE_ELIG: dict[int, str] = {
    mid: elig for mid, _, _, elig in R32_SLOTS if elig is not None
}


def assign_third_place_teams(all_thirds_list: list, rng=None) -> dict:
    """Asigna los 8 mejores terceros a los slots R32 respetando elegibilidad FIFA.

    Retorna {match_number: team}. `rng` se acepta por compatibilidad con la
    simulación pero no se usa (la asignación es determinista por backtracking).
    """
    sorted_thirds = sorted(
        all_thirds_list,
        key=lambda t: (-t["pts"], -t["gd"], -t["gf"], t["team"]),
    )
    third_slots = {m: elig for m, _, _, elig in R32_SLOTS if elig is not None}
    qualified_groups = {t["group"] for t in sorted_thirds[:8]}
    eligible_by_slot = {
        m: [t for t in sorted_thirds[:8] if t["group"] in elig and t["group"] in qualified_groups]
        for m, elig in third_slots.items()
    }
    slots_ordered = sorted(third_slots.keys(), key=lambda m: len(eligible_by_slot[m]))

    def backtrack(slot_idx, used_groups, assignment):
        if slot_idx == len(slots_ordered):
            return assignment
        match_id = slots_ordered[slot_idx]
        for candidate in eligible_by_slot[match_id]:
            if candidate["group"] in used_groups:
                continue
            used_groups.add(candidate["group"])
            assignment[match_id] = candidate["team"]
            result = backtrack(slot_idx + 1, used_groups, assignment)
            if result is not None:
                return result
            used_groups.discard(candidate["group"])
            del assignment[match_id]
        return None

    result = backtrack(0, set(), {})
    if result is not None:
        return result

    assignment, used_groups = {}, set()
    for match_id in slots_ordered:
        for candidate in sorted_thirds:
            if candidate["group"] not in used_groups:
                assignment[match_id] = candidate["team"]
                used_groups.add(candidate["group"])
                break
    return assignment
