from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.base import Base
from app.db.models.match import Match
from app.services.results_sync import _find_match

DT = datetime(2026, 6, 30, 12, 0)


@pytest.fixture
def db():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    with Session() as s:
        yield s


def _third_slot_match(home, away):
    return Match(
        match_number=82, home_team=home, away_team=away,
        home_slot="1G", away_slot="3rd", date=DT,
        stage="Round of 32", status="scheduled",
    )


def test_exact_match_takes_priority(db):
    db.add(_third_slot_match("Belgium", "Senegal"))
    db.commit()
    m, reversed_ = _find_match(db, "Belgium", "Senegal")
    assert m is not None and reversed_ is False
    assert m.away_team == "Senegal"


def test_adopts_real_third_over_wrong_guess(db):
    """La DB tenía un tercero mal (Algeria); el API dice Senegal → se corrige."""
    db.add(_third_slot_match("Belgium", "Algeria"))
    db.commit()
    m, reversed_ = _find_match(db, "Belgium", "Senegal")
    assert m is not None and reversed_ is False
    assert m.away_team == "Senegal"


def test_adopts_third_when_slot_is_null(db):
    db.add(_third_slot_match("Belgium", None))
    db.commit()
    m, _ = _find_match(db, "Belgium", "Senegal")
    assert m is not None
    assert m.away_team == "Senegal"


def test_adopts_third_reversed_orientation(db):
    """API lista al lado fijo como visitante → is_reversed True y adopta el tercero."""
    db.add(_third_slot_match("Belgium", None))
    db.commit()
    m, reversed_ = _find_match(db, "Senegal", "Belgium")
    assert m is not None and reversed_ is True
    assert m.away_team == "Senegal"


def test_no_match_returns_none(db):
    db.add(_third_slot_match("Belgium", "Algeria"))
    db.commit()
    m, _ = _find_match(db, "Germany", "France")
    assert m is None
