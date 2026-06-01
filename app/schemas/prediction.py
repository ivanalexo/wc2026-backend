from pydantic import BaseModel, ConfigDict, computed_field, model_validator

_PREDICTION_LABELS = {
    "H": "Victoria local",
    "D": "Empate",
    "A": "Victoria visitante",
}

class ShapExplanation(BaseModel):
    main_factor: str
    factors: list[str]


class PredictionRequest(BaseModel):
    home_team: str
    away_team: str


class PredictionSummary(BaseModel):
    """Versión reducida para embeds dentro de MatchResponse."""
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

    @model_validator(mode="after")
    def probabilities_sum_to_one(self) -> "PredictionResponse":
        total = round(self.p_home_win + self.p_draw + self.p_away_win, 4)
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Las probabilidades deben sumar 1.0, suman {total}")
        return self

class ScoreEntryResponse(BaseModel):
    """Un marcador posible con su probabilidad estimada."""
    score: str           # "2-1"
    probability: float   # 0.087


class ScorePredictionResponse(BaseModel):
    """
    Respuesta del endpoint GET /predict/score.
    Incluye el marcador más probable, los λ usados, las probabilidades
    de resultado derivadas de la simulación y la distribución de los
    marcadores más frecuentes.
    """
    home_team: str
    away_team: str

    # Marcador más probable (moda de la simulación)
    predicted_home_goals: int
    predicted_away_goals: int

    # Parámetros Poisson usados (útiles para el frontend / debugging)
    expected_home_goals: float
    expected_away_goals: float

    # Probabilidades de resultado derivadas de la simulación Monte Carlo
    # (independientes de las de XGBoost — pueden diferir ligeramente)
    p_home_win: float
    p_draw: float
    p_away_win: float

    # Los marcadores más probables ordenados de mayor a menor probabilidad
    top_scores: list[ScoreEntryResponse]

    n_simulations: int

    @computed_field
    @property
    def predicted_score(self) -> str:
        """Formato legible: '2-1'."""
        return f"{self.predicted_home_goals}-{self.predicted_away_goals}"