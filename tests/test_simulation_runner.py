from datetime import datetime

from app.db.models.match import Match
from app.services.simulation_runner import build_played_results

DT = datetime(2026, 6, 20, 12, 0)


def _m(home, away, hs, as_, stage, status="finished"):
    return Match(home_team=home, away_team=away, home_score=hs, away_score=as_,
                 date=DT, stage=stage, status=status)


def test_played_results_only_includes_group_stage():
    """Los KO no deben entrar a played_results (solo condiciona la fase de grupos)."""
    matches = [
        _m("Mexico", "South Korea", 2, 0, "Group Stage"),
        _m("Brazil", "Spain", 1, 0, "Round of 32"),      # KO: se ignora
    ]
    played = build_played_results(matches)

    assert frozenset(("Mexico", "South Korea")) in played
    assert frozenset(("Brazil", "Spain")) not in played
    assert len(played) == 1
