"""Input validation and preparation helpers."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


OPTIONAL_METADATA_COLUMNS = ["Machine ID", "Timestamp", "Location", "Operator"]
MISSING_VALUE_OPTIONS = ["Drop rows with missing values", "Fill missing values with median"]


def required_columns_missing(df: pd.DataFrame, feature_columns: Iterable[str]) -> list[str]:
    return [column for column in feature_columns if column not in df.columns]


def metadata_columns_present(df: pd.DataFrame) -> list[str]:
    return [column for column in OPTIONAL_METADATA_COLUMNS if column in df.columns]


def dataset_medians(dataset_stats: pd.DataFrame, feature_columns: Iterable[str]) -> dict[str, float]:
    medians = {}
    for feature in feature_columns:
        if feature in dataset_stats.index and "median" in dataset_stats.columns:
            medians[feature] = float(dataset_stats.loc[feature, "median"])
        elif feature in dataset_stats.index and "50%" in dataset_stats.columns:
            medians[feature] = float(dataset_stats.loc[feature, "50%"])
    return medians


def prepare_prediction_dataframe(
    df: pd.DataFrame,
    feature_columns: Iterable[str],
    missing_strategy: str,
    medians: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Prepare uploaded/simulated data while preserving row-level failures."""

    feature_columns = list(feature_columns)
    missing = required_columns_missing(df, feature_columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")

    if missing_strategy not in MISSING_VALUE_OPTIONS:
        raise ValueError(f"Invalid missing value option: {missing_strategy}")

    prepared = df.copy()
    original_missing_mask = prepared[feature_columns].isna()
    original_missing_rows = original_missing_mask.any(axis=1)

    invalid_numeric_cells: dict[str, int] = {}
    for feature in feature_columns:
        source = prepared[feature]
        numeric = pd.to_numeric(source, errors="coerce")
        invalid_mask = source.notna() & numeric.isna()
        if invalid_mask.any():
            invalid_numeric_cells[feature] = int(invalid_mask.sum())
        prepared[feature] = numeric

    dropped_rows = 0
    if missing_strategy == "Drop rows with missing values":
        dropped_rows = int(original_missing_rows.sum())
        prepared = prepared.loc[~original_missing_rows].copy()
    else:
        medians = medians or {}
        for feature in feature_columns:
            fill_value = medians.get(feature, float(prepared[feature].median()))
            original_missing = original_missing_mask.loc[prepared.index, feature]
            prepared.loc[original_missing, feature] = fill_value

    report = {
        "missing_cells": int(original_missing_mask.sum().sum()),
        "rows_with_missing": int(original_missing_rows.sum()),
        "dropped_rows": dropped_rows,
        "invalid_numeric_cells": invalid_numeric_cells,
    }
    return prepared.reset_index(drop=True), report


def out_of_range_report(
    df: pd.DataFrame,
    feature_columns: Iterable[str],
    stats: pd.DataFrame,
) -> list[str]:
    messages: list[str] = []
    for feature in feature_columns:
        if feature not in df.columns or feature not in stats.index:
            continue

        values = pd.to_numeric(df[feature], errors="coerce")
        min_value = float(stats.loc[feature, "min"])
        max_value = float(stats.loc[feature, "max"])
        below = int((values < min_value).sum())
        above = int((values > max_value).sum())

        if below or above:
            messages.append(
                f"{feature}: {below} values are below the dataset range and {above} values are above the dataset range "
                f"({min_value:.2f} to {max_value:.2f})."
            )

    return messages


def summarize_batch_quality(result_df: pd.DataFrame) -> dict[str, int]:
    if result_df.empty or "error_message" not in result_df.columns:
        return {"success": 0, "failed": 0}
    failed = int(result_df["error_message"].fillna("").astype(str).str.len().gt(0).sum())
    success = int(len(result_df) - failed)
    return {"success": success, "failed": failed}


def coerce_manual_payload(payload: dict[str, object], feature_columns: Iterable[str]) -> dict[str, float]:
    coerced = dict(payload)
    for feature in feature_columns:
        value = coerced.get(feature)
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{feature} must be numeric.") from exc
        if not np.isfinite(number):
            raise ValueError(f"{feature} is invalid.")
        coerced[feature] = number
    return coerced
