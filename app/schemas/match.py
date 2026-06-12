from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.schemas.prediction import PredictionSummary

class MatchResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    home_team: str
    away_team: str
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