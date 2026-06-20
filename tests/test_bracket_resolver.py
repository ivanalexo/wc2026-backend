from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.base import Base
from app.db.models.match import Match
from app.services.bracket_resolver import resolve_bracket

DT = datetime(2026, 6, 20, 12, 0)


@pytest.fixture
def db():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    with Session() as s:
        yield s


def _group_match(num, home, away, hs, as_, group="A"):
    return Match(
        match_number=num, home_team=home, away_team=away,
        home_score=hs, away_score=as_, date=DT, group=group,
        stage="Group Stage", status="finished",
    )


def _seed_group_a(db):
    """Group A: MX 1º (9), KR 2º (6), ZA 3º (3), CZ 4º (0)."""
    db.add_all([
        _group_match(1, "Mexico", "South Korea", 2, 0),
        _group_match(2, "Mexico", "South Africa", 2, 0),
        _group_match(3, "Mexico", "Czech Republic", 2, 0),
        _group_match(4, "South Korea", "South Africa", 1, 0),
        _group_match(5, "South Korea", "Czech Republic", 1, 0),
        _group_match(6, "South Africa", "Czech Republic", 1, 0),
    ])
    db.commit()


def test_group_slots_filled_for_resolved_group(db):
    _seed_group_a(db)
    db.add(Match(match_number=73, home_slot="1A", away_slot="2A",
                 date=DT, stage="Round of 32", status="scheduled"))
    db.commit()

    resolve_bracket(db)

    m = db.query(Match).filter(Match.match_number == 73).one()
    assert m.home_team == "Mexico"        # 1A
    assert m.away_team == "South Korea"   # 2A


def test_unresolved_group_leaves_slot_null(db):
    _seed_group_a(db)
    db.add(Match(match_number=73, home_slot="1A", away_slot="2B",  # B no sembrado
                 date=DT, stage="Round of 32", status="scheduled"))
    db.commit()

    resolve_bracket(db)

    m = db.query(Match).filter(Match.match_number == 73).one()
    assert m.home_team == "Mexico"
    assert m.away_team is None            # 2B sin resolver


def test_knockout_winner_and_loser_propagate(db):
    db.add_all([
        Match(match_number=101, home_team="X", away_team="Y",
              home_score=3, away_score=1, date=DT,
              stage="Quarter-finals", status="finished"),
        Match(match_number=90, home_slot="W101", away_slot="L101",
              date=DT, stage="Round of 16", status="scheduled"),
    ])
    db.commit()

    resolve_bracket(db)

    m = db.query(Match).filter(Match.match_number == 90).one()
    assert m.home_team == "X"   # ganador de 101
    assert m.away_team == "Y"   # perdedor de 101


def test_draw_without_penalties_does_not_propagate(db):
    db.add_all([
        Match(match_number=102, home_team="A1", away_team="B1",
              home_score=1, away_score=1, date=DT,
              stage="Quarter-finals", status="finished"),
        Match(match_number=91, home_slot="W102", away_slot=None,
              date=DT, stage="Round of 16", status="scheduled"),
    ])
    db.commit()

    resolve_bracket(db)

    m = db.query(Match).filter(Match.match_number == 91).one()
    assert m.home_team is None   # empate sin penales → no se propaga


def test_resolver_is_idempotent(db):
    _seed_group_a(db)
    db.add(Match(match_number=73, home_slot="1A", away_slot="2A",
                 date=DT, stage="Round of 32", status="scheduled"))
    db.commit()

    first = resolve_bracket(db)
    second = resolve_bracket(db)

    assert first.slots_filled == 2
    assert second.slots_filled == 0          # nada nuevo que escribir
    assert second.knockout_propagated == 0
