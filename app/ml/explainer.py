# =============================================================================
# app/ml/explainer.py — SHAP values → lenguaje natural
# =============================================================================

from __future__ import annotations

import numpy as np
import pandas as pd

from app.ml.loader import MLArtifacts, MODEL_FEATURES, CLASS_ORDER
from app.schemas.prediction import ShapExplanation

_FEATURE_LABELS: dict[str, tuple[str, str]] = {
    "elo_diff":           ("nivel histórico superior",           "nivel histórico inferior"),
    "elo_prob_home":      ("mayor probabilidad ELO",             "menor probabilidad ELO"),
    "is_neutral":         ("terreno neutral",                    "terreno neutral"),
    "home_form_5":        ("buena forma reciente (local)",       "mala forma reciente (local)"),
    "away_form_5":        ("rival en buena forma",               "rival en mala forma"),
    "form_diff_5":        ("mejor forma reciente",               "peor forma reciente"),
    "home_form_10":       ("solidez en últimos 10 partidos",     "irregularidad reciente (local)"),
    "away_form_10":       ("rival consistente",                  "rival inconsistente"),
    "form_diff_10":       ("ventaja de forma acumulada",         "desventaja de forma acumulada"),
    "home_avg_gf":        ("alta potencia ofensiva",             "baja potencia ofensiva"),
    "home_avg_ga":        ("solidez defensiva",                  "vulnerabilidad defensiva"),
    "away_avg_gf":        ("rival ofensivo",                     "rival poco goleador"),
    "away_avg_ga":        ("rival con defensa débil",            "rival con defensa sólida"),
    "gf_diff":            ("superioridad ofensiva",              "inferioridad ofensiva"),
    "ga_diff":            ("superioridad defensiva",             "inferioridad defensiva"),
    "h2h_total":          ("historial de enfrentamientos",       "historial de enfrentamientos"),
    "h2h_home_rate":      ("historial H2H favorable",            "historial H2H desfavorable"),
    "home_penalty_rate":  ("mejor historial en penales",         "peor historial en penales"),
    "away_penalty_rate":  ("rival con buen historial en penales","rival con mal historial en penales"),
}

_CLASS_IDX = {c: i for i, c in enumerate(CLASS_ORDER)}


def _extract_shap_for_class(shap_values, class_idx: int) -> np.ndarray:
    """
    Extrae el vector SHAP para una clase específica.

    SHAP cambia su formato de retorno según la versión:
      - Formato antiguo: lista de arrays, uno por clase
                         shap_values[class_idx] → shape (n_samples, n_features)
      - Formato nuevo:   array 3D único
                         shap_values → shape (n_samples, n_features, n_classes)

    Retorna siempre un vector 1D de shape (n_features,).
    """
    if isinstance(shap_values, list):
        # Formato antiguo: lista de (n_samples, n_features)
        return np.array(shap_values[class_idx][0])

    arr = np.array(shap_values)

    if arr.ndim == 3:
        # Formato nuevo: (n_samples, n_features, n_classes)
        return arr[0, :, class_idx]

    if arr.ndim == 2:
        # Array 2D: (n_samples, n_features) — modelo binario o single output
        return arr[0]

    # Fallback: ya es 1D
    return arr


def explain_prediction(
    features: dict,
    prediction: str,
    home_team: str,
    away_team: str,
    artifacts: MLArtifacts,
    top_n: int = 4,
) -> ShapExplanation:
    """
    Calcula SHAP values para la clase predicha y retorna los top_n
    factores más influyentes en lenguaje natural.
    """
    X = pd.DataFrame([features])[MODEL_FEATURES]
    X_imputed = artifacts.imputer.transform(X)

    shap_values = artifacts.shap_explainer.shap_values(X_imputed)

    class_idx    = _CLASS_IDX.get(prediction, 2)
    shap_for_class = _extract_shap_for_class(shap_values, class_idx)

    # Ordenamos features por valor absoluto descendente
    ranked = sorted(
        zip(MODEL_FEATURES, shap_for_class),
        key=lambda x: abs(x[1]),
        reverse=True,
    )[:top_n]

    factors = []
    for feature, shap_val in ranked:
        labels = _FEATURE_LABELS.get(feature, (feature, feature))
        factors.append(labels[0] if shap_val >= 0 else labels[1])

    # Factor principal como frase completa
    main_feature, main_val = ranked[0]
    main_labels = _FEATURE_LABELS.get(main_feature, (main_feature, main_feature))
    main_label  = main_labels[0] if main_val >= 0 else main_labels[1]

    subject     = home_team if prediction in ("H", "D") else away_team
    main_factor = f"{subject} tiene {main_label}"

    return ShapExplanation(main_factor=main_factor, factors=factors)