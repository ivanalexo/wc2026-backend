from app.services.tiebreaker import GroupMatch, rank_group


def _order(table):
    return [r["team"] for r in table]


def test_clear_ordering():
    """Sin empates: orden por puntos."""
    matches = [
        GroupMatch("A", "B", 1, 0),
        GroupMatch("A", "C", 1, 0),
        GroupMatch("A", "D", 1, 0),
        GroupMatch("B", "C", 1, 0),
        GroupMatch("B", "D", 1, 0),
        GroupMatch("C", "D", 1, 0),
    ]
    assert _order(rank_group(["A", "B", "C", "D"], matches)) == ["A", "B", "C", "D"]


def test_tie_resolved_by_goal_difference():
    """Tres equipos a 6 pts se separan por diferencia de goles global (sin h2h)."""
    matches = [
        GroupMatch("A", "C", 3, 0),
        GroupMatch("A", "D", 3, 0),
        GroupMatch("B", "A", 1, 0),   # A pierde con B
        GroupMatch("B", "C", 1, 0),
        GroupMatch("D", "B", 1, 0),   # B pierde con D
        GroupMatch("D", "C", 2, 0),
    ]
    # A: 6 gd+5 | B: 6 gd+1 | D: 6 gd0 | C: 0
    assert _order(rank_group(["A", "B", "C", "D"], matches)) == ["A", "B", "D", "C"]


def test_two_way_tie_resolved_by_head_to_head():
    """A y B idénticos en pts/dif/GF global; el head-to-head (A venció a B) decide."""
    matches = [
        GroupMatch("A", "B", 1, 0),   # h2h: A > B
        GroupMatch("A", "C", 2, 0),
        GroupMatch("D", "A", 1, 0),
        GroupMatch("B", "C", 2, 0),
        GroupMatch("B", "D", 1, 0),
        GroupMatch("D", "C", 1, 0),
    ]
    # A: 6 gf3 ga1 gd+2 | B: 6 gf3 ga1 gd+2 | D: 6 gf2 ga1 gd+1 | C: 0
    table = _order(rank_group(["A", "B", "C", "D"], matches))
    assert table == ["A", "B", "D", "C"], table


def test_three_way_cycle_falls_back_to_alphabetical():
    """Ciclo perfecto (todo igual incluso en h2h) → fallback determinista alfabético."""
    matches = [
        GroupMatch("B", "A", 1, 0),   # ciclo A<B
        GroupMatch("C", "B", 1, 0),   # B<C
        GroupMatch("A", "C", 1, 0),   # C<A
        GroupMatch("A", "D", 1, 0),
        GroupMatch("B", "D", 1, 0),
        GroupMatch("C", "D", 1, 0),
    ]
    # A, B, C: 6 pts, gd+1, gf2 (idénticos global y en h2h) → alfabético; D último
    assert _order(rank_group(["A", "B", "C", "D"], matches)) == ["A", "B", "C", "D"]
