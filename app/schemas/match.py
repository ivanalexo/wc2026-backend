from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.schemas.prediction import PredictionSummary

class MatchResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    match_number: int | None = None
    # NULL en partidos de eliminatoria aún sin equipo definido; ver home_slot/away_slot.
    home_team: str | None = None
    away_team: str | None = None
    home_slot: str | None = None
    away_slot: str | None = None
    date: datetime
    city: str | None
    country: str | None
    stage: str | None
    group: str | None
    status: str
    home_score: int | None
    away_score: int | None

class MatchWithPrediction(MatchResponse):
    prediction: PredictionSummary | None = None