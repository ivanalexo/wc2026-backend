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

CLASS_ORDER: list[str] = ["H", "D", "A"]


@dataclass
class MLArtifacts:
    imputer: Any
    model: Any
    T_optimal: float
    xgb_metadata: dict
    calibration_table: pd.DataFrame
    bin_edges: list
    bin_labels: list
    shap_explainer: Any
    fixture_features: pd.DataFrame
    montecarlo: pd.DataFrame
    team_elo: dict[str, float]
    team_name_map: dict[str, str]

    # Historial completo de partidos internacionales para H2H entre cualquier par.
    # Es None si clean_results_historical.csv no está en artifacts/ (no rompe el startup).
    historical_results: pd.DataFrame | None


def _patch_imputer(imputer: Any) -> Any:
    if hasattr(imputer, "_fit_dtype") and not hasattr(imputer, "_fill_dtype"):
        imputer._fill_dtype = imputer._fit_dtype
    elif hasattr(imputer, "_fill_dtype") and not hasattr(imputer, "_fit_dtype"):
        imputer._fit_dtype = imputer._fill_dtype
    return imputer


def _build_team_lookups(
    fixture: pd.DataFrame,
) -> tuple[dict[str, float], dict[str, str]]:
    team_elo: dict[str, float] = {}
    team_name_map: dict[str, str] = {}

    for _, row in fixture.iterrows():
        for team_col, elo_col in [("home_team", "home_elo"), ("away_team", "away_elo")]:
            name = row[team_col]
            key  = name.lower()
            team_name_map[key] = name
            if pd.notna(row.get(elo_col)):
                team_elo[key] = float(row[elo_col])

    return team_elo, team_name_map


def load_artifacts() -> MLArtifacts:
    base: Path = settings.artifacts_dir

    def _require(filename: str) -> Path:
        path = base / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Artefacto requerido no encontrado: {path}\n"
                "Copia los outputs de las Fases 1-2 a la carpeta 'artifacts/'."
            )
        return path

    with open(_require("xgb_pipeline.pkl"), "rb") as f:
        xgb_data: dict = pickle.load(f)

    imputer      = _patch_imputer(xgb_data["imputer"])
    model        = xgb_data["model"]
    T_optimal    = float(xgb_data.get("T_optimal", 1.0))
    xgb_metadata = xgb_data.get("metadata", {})

    with open(_require("baseline_model.pkl"), "rb") as f:
        baseline_data: dict = pickle.load(f)

    shap_explainer   = shap.TreeExplainer(model)
    fixture_features = pd.read_csv(_require("master_fixture_2026.csv"), parse_dates=["date"])
    montecarlo       = pd.read_csv(_require("montecarlo_probabilities.csv"))

    team_elo, team_name_map = _build_team_lookups(fixture_features)

    # Historial completo — opcional, no rompe el startup si no existe
    hist_path = base / "clean_results_historical.csv"
    if hist_path.exists():
        historical_results = pd.read_csv(hist_path, parse_dates=["date"])
    else:
        import logging
        logging.getLogger(__name__).warning(
            "clean_results_historical.csv no encontrado en artifacts/. "
            "El endpoint /stats/h2h no estará disponible."
        )
        historical_results = None

    return MLArtifacts(
        imputer=imputer,
        model=model,
        T_optimal=T_optimal,
        xgb_metadata=xgb_metadata,
        calibration_table=baseline_data["calibration_table"],
        bin_edges=baseline_data["bin_edges"],
        bin_labels=baseline_data["bin_labels"],
        shap_explainer=shap_explainer,
        fixture_features=fixture_features,
        montecarlo=montecarlo,
        team_elo=team_elo,
        team_name_map=team_name_map,
        historical_results=historical_results,
    )