import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import shap

from app.config import settings

MODEL_FEATURES: list[str] = [
    "elo_diff", "elo_prob_home",
    "is_neutral",
    "home_form_5", "away_form_5", "form_diff_5",
    "home_form_10", "away_form_10", "form_diff_10",
    "home_avg_gf", "home_avg_ga", "away_avg_gf", "away_avg_ga", "gf_diff", "ga_diff",
    "h2h_total", "h2h_home_rate",
    "home_penalty_rate", "away_penalty_rate",
]

CLASS_ORDER: list[str] = ['A', 'D', 'H']

@dataclass
class MLArtifacts:
    # Pipelien XGBoost
    imputer: Any # Sklearn SimpleImputer
    mode: Any # XGBClassifier entrenado
    T_optimal: float # Temperatura Scaling`
    xgb_metadata: dict

    # Baseline calibrado
    calibration_table: pd.DataFrame
    bin_edges: list
    bin_labels: list

    # SHAP
    shap_explainer: Any # SHAP TreeExplainer

    # Datos pre computados
    fixture_features: pd.DataFrame # 72 partidos con sus 19 features
    montecarlo: pd.DataFrame # probabilidades por rond (48 equipos)

    # Lookup de Elo por equipo
    team_elo: dict[str, float]

def _build_team_elo(fixture: pd.DataFrame) -> dict[str, float]:
    """
    Construye un dict {equipo: elo} a apartir del fixture.
    Si un equipo aparece varias veces, se toma el último elo registrado.
    """
    elo: dict[str, float] = {}
    for _, row in fixture.iterrows():
        if pd.notna(row.get('home_elo')):
            elo[row['home_team']] = float(row['home_elo'])
        if pd.notna(row.get('away_elo')):
            elo[row['away_team']] = float(row['away_elo'])
    return elo

def load_artifacts() -> MLArtifacts:
    base: Path = settings.artifacts_dir

    def _require(filename: str) -> Path:
        path = base / filename
        if not path.exists():
            raise FileNotFoundError(
                f'Artefacto requerido no encontrado: {path}\n'
                'Copia los outputs a la carpeta /artifacts'
            )
        return path
    
    with open(_require('xgb_pipeline.pkl'), 'rb') as f:
        xgb_data: dict = pickle.load(f)
    
    imputer = xgb_data['imputer']
    model = xgb_data['model']
    T_optimal = float(xgb_data.get('T_optimal', 1.0))
    xgb_metadata = xgb_data.get('metadata', {})

    with open(_require("baseline_model.pkl"), "rb") as f:
        baseline_data: dict = pickle.load(f)
 
    calibration_table = baseline_data["calibration_table"]
    bin_edges         = baseline_data["bin_edges"]
    bin_labels        = baseline_data["bin_labels"]

    shap_explainer = shap.TreeExplainer(model)

    fixture_features = pd.read_csv(
        _require('master_fixture_2026.csv'), parse_dates=['date']
    )

    montecarlo = pd.read_csv(_require("montecarlo_probabilities.csv"))
 
    team_elo = _build_team_elo(fixture_features)
 
    return MLArtifacts(
        imputer=imputer,
        model=model,
        T_optimal=T_optimal,
        xgb_metadata=xgb_metadata,
        calibration_table=calibration_table,
        bin_edges=bin_edges,
        bin_labels=bin_labels,
        shap_explainer=shap_explainer,
        fixture_features=fixture_features,
        montecarlo=montecarlo,
        team_elo=team_elo,
    )
 