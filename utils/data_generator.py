"""Dataset-based template and simulation data generation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SCENARIOS = [
    "Normal Operation",
    "High Tool Wear",
    "High Torque",
    "Low Speed High Torque",
    "Mixed Condition",
]


def load_dataset(dataset_path: str | Path = "dataset/ai4i2020.csv") -> pd.DataFrame:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset was not found: {path}")
    return pd.read_csv(path)


def dataset_stats(
    dataset_path: str | Path,
    feature_columns: Iterable[str],
) -> pd.DataFrame:
    df = load_dataset(dataset_path)
    feature_columns = list(feature_columns)
    missing = [feature for feature in feature_columns if feature not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing features: {', '.join(missing)}")

    stats = df[feature_columns].describe(percentiles=[0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]).T
    stats = stats.rename(columns={"50%": "median"})
    return stats


def create_template_csv(feature_columns: Iterable[str]) -> pd.DataFrame:
    feature_columns = list(feature_columns)
    defaults = {
        "Air temperature [K]": 300.1,
        "Process temperature [K]": 310.1,
        "Rotational speed [rpm]": 1503,
        "Torque [Nm]": 40.1,
        "Tool wear [min]": 108,
    }
    row = {
        "Machine ID": "MCH-001",
        "Timestamp": "2026-06-08 08:00:00",
        "Location": "Line A",
        "Operator": "Operator 1",
    }
    row.update({feature: defaults.get(feature, 0) for feature in feature_columns})
    return pd.DataFrame([row])


def _candidate_rows(df: pd.DataFrame, feature_columns: list[str], scenario: str) -> pd.DataFrame:
    q = df[feature_columns].quantile([0.10, 0.25, 0.50, 0.75, 0.90])

    if scenario == "Normal Operation":
        mask = pd.Series(True, index=df.index)
        for feature in feature_columns:
            mask &= df[feature].between(q.loc[0.25, feature], q.loc[0.75, feature])
        return df.loc[mask, feature_columns]

    if scenario == "High Tool Wear":
        return df.loc[df["Tool wear [min]"] >= q.loc[0.75, "Tool wear [min]"], feature_columns]

    if scenario == "High Torque":
        return df.loc[df["Torque [Nm]"] >= q.loc[0.75, "Torque [Nm]"], feature_columns]

    if scenario == "Low Speed High Torque":
        mask = (df["Rotational speed [rpm]"] <= q.loc[0.25, "Rotational speed [rpm]"]) & (
            df["Torque [Nm]"] >= q.loc[0.75, "Torque [Nm]"]
        )
        return df.loc[mask, feature_columns]

    return df[feature_columns]


def _sample_and_jitter(
    source_df: pd.DataFrame,
    feature_columns: list[str],
    amount: int,
    rng: np.random.Generator,
    stats: pd.DataFrame,
) -> pd.DataFrame:
    if source_df.empty:
        source_df = load_dataset()[feature_columns]

    sampled = source_df.sample(n=amount, replace=len(source_df) < amount, random_state=int(rng.integers(0, 2**32 - 1)))
    sampled = sampled.reset_index(drop=True).astype(float)

    for feature in feature_columns:
        std = float(stats.loc[feature, "std"])
        noise_scale = max(std * 0.03, 1e-6)
        sampled[feature] = sampled[feature] + rng.normal(0, noise_scale, size=amount)
        sampled[feature] = sampled[feature].clip(float(stats.loc[feature, "min"]), float(stats.loc[feature, "max"]))

    return sampled


def _format_machine_values(df: pd.DataFrame) -> pd.DataFrame:
    formatted = df.copy()
    if "Air temperature [K]" in formatted:
        formatted["Air temperature [K]"] = formatted["Air temperature [K]"].round(1)
    if "Process temperature [K]" in formatted:
        formatted["Process temperature [K]"] = formatted["Process temperature [K]"].round(1)
    if "Rotational speed [rpm]" in formatted:
        formatted["Rotational speed [rpm]"] = formatted["Rotational speed [rpm]"].round().astype(int)
    if "Torque [Nm]" in formatted:
        formatted["Torque [Nm]"] = formatted["Torque [Nm]"].round(1)
    if "Tool wear [min]" in formatted:
        formatted["Tool wear [min]"] = formatted["Tool wear [min]"].round().astype(int)
    return formatted


def generate_simulation_data(
    dataset_path: str | Path,
    feature_columns: Iterable[str],
    amount: int = 20,
    scenario: str = "Normal Operation",
    random_seed: int = 42,
) -> pd.DataFrame:
    if amount < 1:
        raise ValueError("Simulation record count must be at least 1.")
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown simulation scenario: {scenario}")

    feature_columns = list(feature_columns)
    df = load_dataset(dataset_path)
    stats = dataset_stats(dataset_path, feature_columns)
    rng = np.random.default_rng(random_seed)

    if scenario == "Mixed Condition":
        scenario_pool = [
            "Normal Operation",
            "High Tool Wear",
            "High Torque",
            "Low Speed High Torque",
        ]
        chunks = []
        choices = rng.choice(scenario_pool, size=amount, replace=True)
        for selected in choices:
            candidates = _candidate_rows(df, feature_columns, selected)
            chunks.append(_sample_and_jitter(candidates, feature_columns, 1, rng, stats))
        generated = pd.concat(chunks, ignore_index=True)
    else:
        candidates = _candidate_rows(df, feature_columns, scenario)
        generated = _sample_and_jitter(candidates, feature_columns, amount, rng, stats)

    generated = _format_machine_values(generated)
    generated.insert(0, "Machine ID", [f"SIM-{index + 1:03d}" for index in range(len(generated))])
    generated.insert(
        1,
        "Timestamp",
        pd.date_range("2026-06-09 08:00:00", periods=len(generated), freq="min").strftime("%Y-%m-%d %H:%M:%S"),
    )
    generated.insert(2, "Location", "Simulation Line")
    generated.insert(3, "Operator", "Demo Operator")
    return generated
