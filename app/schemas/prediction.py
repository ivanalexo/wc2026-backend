from pydantic import BaseModel, ConfigDict, computed_field, model_validator

_PREDICTION_LABELS = {
    'H': 'Victoria local',
    'D': 'Empate',
    'A': 'Victoria visitante'
}

class ShapExplanation(BaseModel):
    """Explicacion de la prediccion en lenguaje natural, derivada de SHAP """

    main_factor: str
    factors: list[str]

class PredictionRequest(BaseModel):
    home_team: str
    away_team: str

class PredictionSummary(BaseModel):
    p_home_win: float
    p_draw: float
    p_away_win: float
    prediction: str
    prediction_label: str

class PredictionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    home_team: str
    away_team: str
    p_home_win: float
    p_draw: float
    p_away_win: float
    prediction: str
    home_elo: float | None
    away_elo: float | None
    elo_diff: float | None
    explanation: ShapExplanation | None = None
    cached: bool = False

    @computed_field
    @property
    def prediction_label(self) -> str:
        return _PREDICTION_LABELS.get(self.prediction, self.prediction)
    
    @model_validator(mode='after')
    def probabilities_sum_to_one(self) -> 'PredictionResponse':
        total = round(self.p_home_win + self.p_draw + self.p_away_win, 4)
        if abs(total - 1.0) > 0.01:
            raise ValueError(f'Las probabilidades deben sumar 1.0, pero suman {total}')
        return self