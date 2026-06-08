"""ANFIS inference utilities.

The inference formula mirrors the project notebook:
scaled input -> Gaussian memberships -> linear consequent layer -> sigmoid
risk score -> threshold-based class.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


REQUIRED_MODEL_KEYS = {"centers", "sigmas", "weights", "bias", "threshold", "feature_columns"}
MEMBERSHIP_LABELS = ("Low", "Medium", "High")
STATUS_NORMAL = "Normal"
STATUS_FAILURE = "Failure Risk"


class ArtifactError(RuntimeError):
    """Raised when model artifacts cannot be loaded or validated."""


@dataclass(frozen=True)
class ModelArtifacts:
    scaler: Any
    feature_columns: list[str]
    model: dict[str, Any]
    comparison_model: dict[str, Any] | None = None


def load_pickle(path: str | Path) -> Any:
    """Load a joblib/pickle artifact and raise a clear app-facing error."""

    artifact_path = Path(path)
    if not artifact_path.exists():
        raise ArtifactError(f"Artifact file was not found: {artifact_path}")

    try:
        return joblib.load(artifact_path)
    except Exception as exc:  # pragma: no cover - message is user-facing
        raise ArtifactError(f"Failed to read artifact {artifact_path.name}: {exc}") from exc


def _validate_model_dict(model: Any, model_name: str) -> dict[str, Any]:
    if not isinstance(model, dict):
        raise ArtifactError(
            f"{model_name} must be a dictionary. Loaded type: {type(model).__name__}."
        )

    missing_keys = sorted(REQUIRED_MODEL_KEYS - set(model.keys()))
    if missing_keys:
        available = ", ".join(map(str, model.keys()))
        missing = ", ".join(missing_keys)
        raise ArtifactError(
            f"{model_name} is missing required keys: {missing}. Available keys: {available}"
        )

    centers = np.asarray(model["centers"], dtype=float)
    sigmas = np.asarray(model["sigmas"], dtype=float)
    weights = np.asarray(model["weights"], dtype=float)
    feature_columns = list(model["feature_columns"])

    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ArtifactError(
            f"{model_name} has an invalid centers shape: {centers.shape}. "
            "Expected (n_features, 3)."
        )
    if sigmas.shape != centers.shape:
        raise ArtifactError(
            f"{model_name} has sigmas shape {sigmas.shape}, which does not match centers {centers.shape}."
        )
    if weights.ndim != 1 or weights.shape[0] != centers.shape[0] * centers.shape[1]:
        raise ArtifactError(
            f"{model_name} has weights shape {weights.shape}. "
            f"Expected ({centers.shape[0] * centers.shape[1]},)."
        )
    if len(feature_columns) != centers.shape[0]:
        raise ArtifactError(
            f"{model_name} has {len(feature_columns)} feature_columns, "
            f"but centers contains {centers.shape[0]} features."
        )

    threshold = float(np.asarray(model["threshold"]).item())
    if not np.isfinite(threshold):
        raise ArtifactError(f"{model_name} has an invalid threshold: {model['threshold']!r}")

    normalized = dict(model)
    normalized["centers"] = centers
    normalized["sigmas"] = sigmas
    normalized["weights"] = weights
    normalized["bias"] = float(np.asarray(model["bias"]).item())
    normalized["threshold"] = threshold
    normalized["feature_columns"] = feature_columns
    return normalized


def load_artifacts(
    models_dir: str | Path = "models",
    scaler_name: str = "scaler.pkl",
    feature_columns_name: str = "feature_columns.pkl",
    main_model_name: str = "anfis_with_ga_params.pkl",
    comparison_model_name: str = "anfis_without_ga_params.pkl",
) -> ModelArtifacts:
    """Load scaler, feature list, main ANFIS+GA model, and optional comparator."""

    models_path = Path(models_dir)
    scaler = load_pickle(models_path / scaler_name)
    feature_columns = list(load_pickle(models_path / feature_columns_name))
    model = _validate_model_dict(load_pickle(models_path / main_model_name), main_model_name)

    comparison_model: dict[str, Any] | None = None
    comparison_path = models_path / comparison_model_name
    if comparison_path.exists():
        comparison_model = _validate_model_dict(load_pickle(comparison_path), comparison_model_name)

    model_features = list(model["feature_columns"])
    if feature_columns != model_features:
        raise ArtifactError(
            "feature_columns.pkl order does not match feature_columns in the main model. "
            f"feature_columns.pkl={feature_columns}; model={model_features}"
        )

    scaler_features = list(getattr(scaler, "feature_names_in_", feature_columns))
    if scaler_features != feature_columns:
        raise ArtifactError(
            "Scaler feature order does not match feature_columns.pkl. "
            f"scaler={scaler_features}; feature_columns={feature_columns}"
        )

    return ModelArtifacts(
        scaler=scaler,
        feature_columns=feature_columns,
        model=model,
        comparison_model=comparison_model,
    )


def sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    """Sigmoid function exactly as used in the notebook."""

    clipped = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-clipped))


def gaussian_membership(x: np.ndarray | float, center: np.ndarray | float, sigma: np.ndarray | float):
    """Gaussian membership function from the notebook."""

    safe_sigma = np.maximum(sigma, 1e-6)
    return np.exp(-((x - center) ** 2) / (2 * safe_sigma**2))


def fuzzify_input(
    scaled_values: np.ndarray,
    centers: np.ndarray,
    sigmas: np.ndarray,
) -> np.ndarray:
    """Convert scaled features into flattened fuzzy values."""

    X_input = np.asarray(scaled_values, dtype=float)
    if X_input.ndim == 1:
        X_input = X_input.reshape(1, -1)

    n_samples = X_input.shape[0]
    n_features = X_input.shape[1]
    n_memberships = centers.shape[1]
    fuzzy_features = np.zeros((n_samples, n_features * n_memberships))

    column_index = 0
    for feature_index in range(n_features):
        for membership_index in range(n_memberships):
            fuzzy_features[:, column_index] = gaussian_membership(
                X_input[:, feature_index],
                centers[feature_index, membership_index],
                sigmas[feature_index, membership_index],
            )
            column_index += 1

    return fuzzy_features


def _status_from_prediction(prediction: int) -> str:
    return STATUS_FAILURE if prediction == 1 else STATUS_NORMAL


def recommendation_for_status(status: str, top_factors: list[str] | None = None) -> str:
    if status == STATUS_FAILURE:
        if top_factors:
            factors = ", ".join(top_factors[:2])
            return (
                "The machine is predicted to have a failure risk. Perform a technical inspection "
                f"on the most influential factors, especially {factors}. The final decision "
                "must still be validated by a technician/operator."
            )
        return (
            "The machine is predicted to have a failure risk. Perform a technical inspection on the "
            "most influential factors. The final decision must still be validated by a technician/operator."
        )

    return "The machine is predicted to be in normal condition. Continue periodic monitoring according to procedure."


def _prepare_single_row(
    row: pd.Series | dict[str, Any],
    feature_columns: list[str],
) -> pd.DataFrame:
    source = pd.Series(row).copy()
    missing_features = [feature for feature in feature_columns if feature not in source.index]
    if missing_features:
        raise ValueError(f"Required feature columns are missing: {', '.join(missing_features)}")

    feature_values = {}
    for feature in feature_columns:
        try:
            value = float(source[feature])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Feature value {feature} must be numeric.") from exc

        if not np.isfinite(value):
            raise ValueError(f"Feature value {feature} cannot be empty or invalid.")
        feature_values[feature] = value

    return pd.DataFrame([feature_values], columns=feature_columns)


def calculate_feature_contributions(
    fuzzy_values: np.ndarray,
    weights: np.ndarray,
    feature_columns: list[str],
) -> dict[str, Any]:
    """Calculate safe feature-level contributions when the weight shape supports it."""

    n_features = len(feature_columns)
    if fuzzy_values.ndim != 1 or weights.ndim != 1 or fuzzy_values.shape[0] != weights.shape[0]:
        return {"safe": False, "message": "Fuzzy values and weights shapes do not match."}
    if weights.shape[0] != n_features * 3:
        return {"safe": False, "message": "Weight count does not match 3 memberships per feature."}

    component_values = fuzzy_values * weights
    rows = []
    feature_totals = []

    index = 0
    for feature in feature_columns:
        total = 0.0
        for membership in MEMBERSHIP_LABELS:
            fuzzy_value = float(fuzzy_values[index])
            weight = float(weights[index])
            contribution = float(component_values[index])
            total += contribution
            rows.append(
                {
                    "Feature": feature,
                    "Membership": membership,
                    "Fuzzy Value": fuzzy_value,
                    "Weight": weight,
                    "Contribution": contribution,
                }
            )
            index += 1

        feature_totals.append(
            {
                "Feature": feature,
                "Contribution": float(total),
                "Impact": float(abs(total)),
            }
        )

    totals_df = pd.DataFrame(feature_totals).sort_values("Impact", ascending=False)
    return {
        "safe": True,
        "component_table": pd.DataFrame(rows),
        "feature_table": totals_df,
        "top_factors": totals_df["Feature"].head(3).tolist(),
    }


def predict_single(row: pd.Series | dict[str, Any], artifacts: ModelArtifacts) -> dict[str, Any]:
    """Predict one machine row using the ANFIS+GA model."""

    feature_columns = artifacts.feature_columns
    model = artifacts.model
    original_df = _prepare_single_row(row, feature_columns)

    try:
        scaled_array = artifacts.scaler.transform(original_df)
    except Exception as exc:
        raise ValueError(f"Failed to scale input: {exc}") from exc

    centers = model["centers"]
    sigmas = model["sigmas"]
    weights = model["weights"]
    bias = model["bias"]
    threshold = model["threshold"]

    fuzzy_array = fuzzify_input(scaled_array, centers, sigmas)
    linear_output = np.dot(fuzzy_array, weights) + bias
    risk_score = float(np.asarray(sigmoid(linear_output)).reshape(-1)[0])
    prediction = int(risk_score >= threshold)
    status = _status_from_prediction(prediction)
    decision_margin = float(abs(risk_score - threshold))

    membership_rows = []
    fuzzy_flat = fuzzy_array.reshape(-1)
    index = 0
    for feature_index, feature in enumerate(feature_columns):
        degrees = {}
        for membership in MEMBERSHIP_LABELS:
            degrees[membership] = float(fuzzy_flat[index])
            index += 1
        dominant_membership = max(degrees, key=degrees.get)
        membership_rows.append(
            {
                "Feature": feature,
                "Original Value": float(original_df.iloc[0][feature]),
                "Scaled Value": float(scaled_array[0, feature_index]),
                "Low": degrees["Low"],
                "Medium": degrees["Medium"],
                "High": degrees["High"],
                "Dominant Membership": dominant_membership,
            }
        )

    contribution_info = calculate_feature_contributions(fuzzy_flat, weights, feature_columns)
    top_factors = contribution_info.get("top_factors", []) if contribution_info.get("safe") else []

    return {
        "risk_score": risk_score,
        "threshold": float(threshold),
        "prediction": prediction,
        "status": status,
        "decision_margin": decision_margin,
        "recommendation": recommendation_for_status(status, top_factors),
        "original_input": original_df.iloc[0].to_dict(),
        "scaled_input": {
            feature: float(scaled_array[0, feature_index])
            for feature_index, feature in enumerate(feature_columns)
        },
        "membership_table": pd.DataFrame(membership_rows),
        "fuzzy_values": fuzzy_flat,
        "linear_output": float(linear_output.reshape(-1)[0]),
        "contribution_info": contribution_info,
        "top_factors": top_factors,
    }


def predict_batch(
    df: pd.DataFrame,
    artifacts: ModelArtifacts,
    metadata_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[int, dict[str, Any]]]:
    """Predict each row independently; failing rows keep an error_message."""

    metadata_columns = metadata_columns or []
    rows = []
    details: dict[int, dict[str, Any]] = {}

    for result_index, (_, source_row) in enumerate(df.iterrows()):
        output_row = source_row.to_dict()
        try:
            result = predict_single(source_row, artifacts)
            top_factors = result.get("top_factors", [])
            output_row.update(
                {
                    "risk_score": result["risk_score"],
                    "threshold": result["threshold"],
                    "prediction": result["prediction"],
                    "status": result["status"],
                    "decision_margin": result["decision_margin"],
                    "top_factor_1": top_factors[0] if len(top_factors) > 0 else "",
                    "top_factor_2": top_factors[1] if len(top_factors) > 1 else "",
                    "top_factor_3": top_factors[2] if len(top_factors) > 2 else "",
                    "error_message": "",
                }
            )
            details[result_index] = result
        except Exception as exc:
            output_row.update(
                {
                    "risk_score": np.nan,
                    "threshold": float(artifacts.model.get("threshold", np.nan)),
                    "prediction": np.nan,
                    "status": "",
                    "decision_margin": np.nan,
                    "top_factor_1": "",
                    "top_factor_2": "",
                    "top_factor_3": "",
                    "error_message": str(exc),
                }
            )
        rows.append(output_row)

    result_df = pd.DataFrame(rows)
    ordered_output_columns = (
        [column for column in metadata_columns if column in result_df.columns]
        + [column for column in artifacts.feature_columns if column in result_df.columns]
        + [
            "risk_score",
            "threshold",
            "prediction",
            "status",
            "decision_margin",
            "top_factor_1",
            "top_factor_2",
            "top_factor_3",
            "error_message",
        ]
    )
    remaining_columns = [column for column in result_df.columns if column not in ordered_output_columns]
    return result_df[ordered_output_columns + remaining_columns], details
