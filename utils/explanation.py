"""User-facing explanation helpers for ANFIS predictions."""

from __future__ import annotations

from typing import Any

import pandas as pd


def format_membership_table(prediction_detail: dict[str, Any]) -> pd.DataFrame:
    table = prediction_detail.get("membership_table", pd.DataFrame()).copy()
    numeric_columns = ["Original Value", "Scaled Value", "Low", "Medium", "High"]
    for column in numeric_columns:
        if column in table.columns:
            table[column] = table[column].astype(float).round(4)
    return table


def membership_long_format(prediction_detail: dict[str, Any]) -> pd.DataFrame:
    table = prediction_detail.get("membership_table", pd.DataFrame()).copy()
    if table.empty:
        return pd.DataFrame(columns=["Feature", "Membership", "Degree"])

    long_df = table.melt(
        id_vars=["Feature"],
        value_vars=["Low", "Medium", "High"],
        var_name="Membership",
        value_name="Degree",
    )
    long_df["Degree"] = long_df["Degree"].astype(float)
    return long_df


def dominant_membership_table(prediction_detail: dict[str, Any]) -> pd.DataFrame:
    table = prediction_detail.get("membership_table", pd.DataFrame()).copy()
    if table.empty:
        return pd.DataFrame(columns=["Feature", "Dominant Membership"])
    return table[["Feature", "Dominant Membership"]].copy()


def contribution_table(prediction_detail: dict[str, Any]) -> pd.DataFrame | None:
    contribution_info = prediction_detail.get("contribution_info", {})
    if not contribution_info.get("safe"):
        return None
    table = contribution_info.get("feature_table", pd.DataFrame()).copy()
    if table.empty:
        return None
    table["Contribution"] = table["Contribution"].astype(float)
    table["Impact"] = table["Impact"].astype(float)
    return table


def top_factors(prediction_detail: dict[str, Any], limit: int = 3) -> list[str]:
    factors = prediction_detail.get("top_factors") or []
    return list(factors[:limit])


def build_prediction_narrative(prediction_detail: dict[str, Any]) -> str:
    status = prediction_detail.get("status", "Normal")
    risk_score = float(prediction_detail.get("risk_score", 0.0))
    threshold = float(prediction_detail.get("threshold", 0.0))
    factors = top_factors(prediction_detail, 3)

    if factors:
        factor_text = ", ".join(factors[:2])
        factor_sentence = f" The most contributing factors are {factor_text}."
    else:
        factor_sentence = (
            " The fuzzy membership table can be used to understand the input pattern for each feature."
        )

    if status == "Failure Risk":
        return (
            "Based on the model, this machine record is predicted as Failure Risk because "
            f"the risk score {risk_score:.4f} is above or equal to the model threshold "
            f"{threshold:.4f}.{factor_sentence} This result should still be validated by "
            "a technician/operator before any maintenance action is taken."
        )

    return (
        "Based on the model, this machine record is predicted as Normal because "
        f"the risk score {risk_score:.4f} is below the model threshold {threshold:.4f}."
        f"{factor_sentence} Periodic monitoring is still recommended according to operating procedure."
    )


def contribution_note(prediction_detail: dict[str, Any]) -> str | None:
    contribution_info = prediction_detail.get("contribution_info", {})
    if contribution_info.get("safe"):
        return None
    return (
        "Numeric contribution is not shown because the model parameter structure does not support "
        "safe direct interpretation."
    )
