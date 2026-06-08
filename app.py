from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from utils.data_generator import (
    SCENARIOS,
    create_template_csv,
    dataset_stats,
    generate_simulation_data,
)
from utils.explanation import (
    build_prediction_narrative,
    contribution_note,
    contribution_table,
    format_membership_table,
    membership_long_format,
)
from utils.inference import ArtifactError, load_artifacts, predict_batch, predict_single
from utils.ui_style import apply_custom_css
from utils.validation import (
    MISSING_VALUE_OPTIONS,
    dataset_medians,
    metadata_columns_present,
    out_of_range_report,
    prepare_prediction_dataframe,
    summarize_batch_quality,
)


ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "dataset" / "ai4i2020.csv"
MODE_OPTIONS = ["Manual Input", "CSV / Simulation"]
DATA_SOURCE_OPTIONS = ["Upload CSV", "Generate Simulation"]
PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}
FEATURE_TOOLTIPS = {
    "Air temperature [K]": "Ambient air temperature around the machine, measured in Kelvin.",
    "Process temperature [K]": "Process-side operating temperature, measured in Kelvin.",
    "Rotational speed [rpm]": "Shaft or spindle rotational speed in revolutions per minute.",
    "Torque [Nm]": "Mechanical torque load applied to the machine.",
    "Tool wear [min]": "Estimated accumulated tool wear time in minutes.",
}


st.set_page_config(
    page_title="Industrial Machine Failure Prediction",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)
apply_custom_css()


@st.cache_resource(show_spinner=False)
def get_artifacts():
    return load_artifacts(ROOT / "models")


@st.cache_data(show_spinner=False)
def get_dataset_stats(feature_columns: tuple[str, ...]) -> pd.DataFrame:
    return dataset_stats(DATASET_PATH, feature_columns)


def init_session_state() -> None:
    defaults = {
        "input_mode": "Manual Input",
        "data_source": "Upload CSV",
        "latest_prediction": None,
        "all_predictions": None,
        "uploaded_data": None,
        "simulated_data": None,
        "simulated_csv_bytes": None,
        "batch_predictions": None,
        "selected_prediction": None,
        "prediction_details": {},
        "result_mode": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_header() -> None:
    st.markdown(
        """
        <div class="app-header">
            <h1>Industrial Machine Failure Prediction</h1>
            <p>
                Enter machine operating data to predict whether the machine is in a normal
                condition or at failure risk based on the ANFIS + Genetic Algorithm model.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_rule() -> None:
    st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)


def section_heading(title: str, subtitle: str | None = None) -> None:
    subtitle_html = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(
        f"""
        <div class="section-heading">
            <h2>{title}</h2>
            {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_grid(metrics: list[tuple[str, str, str | None]]) -> None:
    cards = []
    for label, value, badge_status in metrics:
        if badge_status == "Normal":
            value_html = f'<span class="status-badge status-normal">{value}</span>'
        elif badge_status == "Failure Risk":
            value_html = f'<span class="status-badge status-failure">{value}</span>'
        else:
            value_html = f'<div class="metric-value">{value}</div>'
        cards.append(
            f'<div class="metric-card">'
            f'<div class="metric-label">{label}</div>'
            f"{value_html}"
            f"</div>"
        )
    st.markdown(f'<div class="metric-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def table_height(row_count: int, max_height: int = 420, min_height: int = 120) -> int:
    return min(max_height, max(min_height, 44 + max(row_count, 1) * 38))


def render_static_table(df: pd.DataFrame) -> None:
    st.markdown(
        f'<div class="table-scroll"><div class="static-table">{df.to_html(index=False, escape=True)}</div></div>',
        unsafe_allow_html=True,
    )


def render_table(df: pd.DataFrame, max_height: int = 420, hide_index: bool = True) -> None:
    if len(df) <= 12:
        render_static_table(df)
        return

    st.dataframe(
        df,
        width="stretch",
        height=table_height(len(df), max_height=max_height),
        hide_index=hide_index,
    )


def show_warning_box(messages: list[str]) -> None:
    if not messages:
        return
    message_html = "<br>".join(messages)
    st.markdown(f'<div class="warning-box">{message_html}</div>', unsafe_allow_html=True)


def set_manual_result(prediction: dict[str, Any]) -> None:
    result_df = pd.DataFrame(
        [
            {
                **prediction.get("metadata", {}),
                **prediction["original_input"],
                "risk_score": prediction["risk_score"],
                "threshold": prediction["threshold"],
                "prediction": prediction["prediction"],
                "status": prediction["status"],
                "decision_margin": prediction["decision_margin"],
                "top_factor_1": prediction.get("top_factors", [""])[0]
                if prediction.get("top_factors")
                else "",
                "top_factor_2": prediction.get("top_factors", ["", ""])[1]
                if len(prediction.get("top_factors", [])) > 1
                else "",
                "top_factor_3": prediction.get("top_factors", ["", "", ""])[2]
                if len(prediction.get("top_factors", [])) > 2
                else "",
                "error_message": "",
            }
        ]
    )
    st.session_state.latest_prediction = prediction
    st.session_state.batch_predictions = None
    st.session_state.prediction_details = {0: prediction}
    st.session_state.selected_prediction = 0
    st.session_state.result_mode = "manual"
    st.session_state.all_predictions = result_df


def set_batch_result(
    result_df: pd.DataFrame,
    details: dict[int, dict[str, Any]],
    mode: str,
) -> None:
    st.session_state.latest_prediction = None
    st.session_state.batch_predictions = result_df
    st.session_state.prediction_details = details
    st.session_state.result_mode = mode
    st.session_state.all_predictions = result_df
    st.session_state.selected_prediction = next(iter(details.keys()), None)
    st.session_state.download_csv_bytes = result_df.to_csv(index=False).encode("utf-8")


def manual_input_area(artifacts, stats: pd.DataFrame) -> None:
    medians = dataset_medians(stats, artifacts.feature_columns)

    with st.form("manual_prediction_form", clear_on_submit=False, border=False):
        st.text_input(
            "Machine ID",
            value="MCH-001",
            key="manual_machine_id",
            help="Optional identifier used only to label the prediction result.",
        )
        col_a, col_b = st.columns(2)
        values: dict[str, float] = {}
        for index, feature in enumerate(artifacts.feature_columns):
            target_col = col_a if index % 2 == 0 else col_b
            with target_col:
                values[feature] = st.number_input(
                    feature,
                    value=float(medians.get(feature, 0.0)),
                    min_value=None,
                    max_value=None,
                    step=1.0 if "rpm" in feature or "wear" in feature.lower() else 0.1,
                    format="%.1f" if "rpm" not in feature and "wear" not in feature.lower() else "%.0f",
                    help=FEATURE_TOOLTIPS.get(feature),
                )

        submitted = st.form_submit_button("Run Prediction", width="stretch")

    payload = {"Machine ID": st.session_state.get("manual_machine_id", "")}
    payload.update(values)
    show_warning_box(out_of_range_report(pd.DataFrame([payload]), artifacts.feature_columns, stats))

    if submitted:
        try:
            prediction = predict_single(payload, artifacts)
            prediction["metadata"] = {"Machine ID": payload.get("Machine ID", "")}
            set_manual_result(prediction)
            st.success("Prediction completed successfully.")
        except Exception as exc:
            st.error(f"Manual prediction failed: {exc}")


def batch_input_area(artifacts, stats: pd.DataFrame) -> None:
    template_df = create_template_csv(artifacts.feature_columns)
    st.download_button(
        "Download CSV Template",
        data=template_df.to_csv(index=False).encode("utf-8"),
        file_name="machine_prediction_template.csv",
        mime="text/csv",
        width="stretch",
        help="Download a ready-to-use CSV with the required model columns.",
    )

    source_mode = st.selectbox(
        "Data source",
        DATA_SOURCE_OPTIONS,
        key="data_source",
        help="Choose whether to upload actual machine records or generate demo data from dataset statistics.",
    )
    missing_strategy = st.selectbox(
        "Missing value handling",
        MISSING_VALUE_OPTIONS,
        index=0,
        help="Choose how empty numeric feature values should be handled before batch prediction.",
    )

    active_df: pd.DataFrame | None = None
    active_mode = "batch"

    if source_mode == "Generate Simulation":
        col_a, col_b = st.columns(2)
        with col_a:
            amount = st.number_input(
                "Number of records",
                min_value=1,
                max_value=500,
                value=20,
                step=1,
                help="Number of simulated machine records to generate for demonstration.",
            )
        with col_b:
            scenario = st.selectbox(
                "Simulation scenario",
                SCENARIOS,
                index=0,
                help="Scenario pattern used to sample realistic values from dataset statistics.",
            )

        st.markdown(
            '<p class="muted-note">Simulation data is for application demo only and is not actual field data.</p>',
            unsafe_allow_html=True,
        )

        if st.button("Generate Simulation Data", width="stretch"):
            try:
                generated_seed = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000) % 1_000_000
                st.session_state.simulated_data = generate_simulation_data(
                    DATASET_PATH,
                    artifacts.feature_columns,
                    amount=int(amount),
                    scenario=scenario,
                    random_seed=generated_seed,
                )
                st.session_state.simulated_csv_bytes = st.session_state.simulated_data.to_csv(index=False).encode("utf-8")
                st.success("Simulation data generated successfully.")
            except Exception as exc:
                st.error(f"Failed to generate simulation data: {exc}")

        if st.session_state.simulated_data is not None:
            active_df = st.session_state.simulated_data
            active_mode = "simulation"
            st.caption("Simulation data preview")
            render_table(active_df.head(10), max_height=360)
            st.download_button(
                "Download Generated Simulation Data",
                data=st.session_state.get("simulated_csv_bytes", active_df.to_csv(index=False).encode("utf-8")),
                file_name="generated_simulation_data.csv",
                mime="text/csv",
                width="stretch",
                key="download_generated_simulation_data",
                on_click="ignore",
            )
    else:
        uploaded_file = st.file_uploader(
            "Upload CSV",
            type=["csv"],
            accept_multiple_files=False,
            help="Upload a CSV containing the five required machine input features.",
        )

        if uploaded_file is not None:
            try:
                uploaded_df = pd.read_csv(uploaded_file)
                st.session_state.uploaded_data = uploaded_df
                active_df = uploaded_df
                st.caption("First 10 rows preview")
                render_table(uploaded_df.head(10), max_height=360)
            except Exception as exc:
                st.error(f"Failed to read CSV: {exc}")
                return

    if active_df is not None:
        try:
            missing_columns = [feature for feature in artifacts.feature_columns if feature not in active_df.columns]
            if missing_columns:
                st.error(f"Missing required columns: {', '.join(missing_columns)}")
                return

            medians = dataset_medians(stats, artifacts.feature_columns)
            prepared_df, validation_report = prepare_prediction_dataframe(
                active_df,
                artifacts.feature_columns,
                missing_strategy,
                medians=medians,
            )

            if validation_report["missing_cells"]:
                st.warning(
                    f"Found {validation_report['missing_cells']} missing values across "
                    f"{validation_report['rows_with_missing']} rows. "
                    f"Dropped rows: {validation_report['dropped_rows']}."
                )
            if validation_report["invalid_numeric_cells"]:
                invalid_text = ", ".join(
                    f"{feature}: {count}" for feature, count in validation_report["invalid_numeric_cells"].items()
                )
                st.error(
                    "Some feature values are not numeric. Related rows will keep an error_message "
                    f"in the batch result: {invalid_text}"
                )

            show_warning_box(out_of_range_report(prepared_df, artifacts.feature_columns, stats))

            if st.button("Run Batch Prediction", width="stretch"):
                metadata = metadata_columns_present(prepared_df)
                result_df, details = predict_batch(prepared_df, artifacts, metadata_columns=metadata)
                set_batch_result(result_df, details, mode=active_mode)
                quality = summarize_batch_quality(result_df)
                st.success(
                    f"Batch prediction completed. Successful rows: {quality['success']}, failed rows: {quality['failed']}."
                )
        except Exception as exc:
            st.error(f"Failed to process data: {exc}")
    elif source_mode == "Upload CSV":
        st.markdown(
            '<p class="muted-note">Required columns: Air temperature [K], Process temperature [K], '
            'Rotational speed [rpm], Torque [Nm], Tool wear [min].</p>',
            unsafe_allow_html=True,
        )


def render_input_area(artifacts, stats: pd.DataFrame) -> None:
    section_rule()
    section_heading("Machine Data Input", "Choose the most practical way to provide machine operating data.")
    st.segmented_control(
        "Input mode",
        MODE_OPTIONS,
        key="input_mode",
        selection_mode="single",
        label_visibility="collapsed",
    )

    if st.session_state.input_mode == "Manual Input":
        manual_input_area(artifacts, stats)
    else:
        batch_input_area(artifacts, stats)


def render_manual_result(prediction: dict[str, Any]) -> None:
    machine_id = str(prediction.get("metadata", {}).get("Machine ID", "")).strip()
    if machine_id:
        st.caption(f"Machine ID: {machine_id}")
    metric_grid(
        [
            ("Risk Score", f"{prediction['risk_score']:.4f}", None),
            ("Threshold", f"{prediction['threshold']:.4f}", None),
            ("Status", prediction["status"], prediction["status"]),
            ("Decision Margin", f"{prediction['decision_margin']:.4f}", None),
        ]
    )
    st.markdown(
        f'<div class="recommendation-box">{prediction["recommendation"]}</div>',
        unsafe_allow_html=True,
    )


def render_batch_charts(result_df: pd.DataFrame) -> None:
    successful = result_df[result_df["error_message"].fillna("").astype(str).str.len() == 0].copy()
    if successful.empty:
        return

    chart_col_a, chart_col_b = st.columns(2)
    with chart_col_a:
        status_counts = successful["status"].value_counts().rename_axis("Status").reset_index(name="Count")
        fig_status = px.bar(
            status_counts,
            x="Status",
            y="Count",
            color="Status",
            color_discrete_map={"Normal": "#16A34A", "Failure Risk": "#DC2626"},
        )
        fig_status.update_layout(
            title="Prediction Status Count",
            showlegend=False,
            margin=dict(l=10, r=10, t=42, b=10),
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
        )
        st.plotly_chart(fig_status, config=PLOTLY_CONFIG)

    with chart_col_b:
        fig_hist = px.histogram(
            successful,
            x="risk_score",
            nbins=18,
            color_discrete_sequence=["#111827"],
        )
        fig_hist.update_layout(
            title="Risk Score Distribution",
            xaxis_title="Risk Score",
            yaxis_title="Count",
            margin=dict(l=10, r=10, t=42, b=10),
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
        )
        st.plotly_chart(fig_hist, config=PLOTLY_CONFIG)


def render_batch_result(result_df: pd.DataFrame) -> None:
    quality = summarize_batch_quality(result_df)
    successful = result_df[result_df["error_message"].fillna("").astype(str).str.len() == 0].copy()

    total_data = len(result_df)
    normal_count = int((successful["status"] == "Normal").sum()) if not successful.empty else 0
    failure_count = int((successful["status"] == "Failure Risk").sum()) if not successful.empty else 0
    average_risk = float(successful["risk_score"].mean()) if not successful.empty else np.nan
    max_risk = float(successful["risk_score"].max()) if not successful.empty else np.nan

    metric_grid(
        [
            ("Total Records", f"{total_data}", None),
            ("Normal", f"{normal_count}", None),
            ("Failure Risk", f"{failure_count}", None),
            ("Average Risk", "n/a" if np.isnan(average_risk) else f"{average_risk:.4f}", None),
        ]
    )

    extra_col_a, extra_col_b = st.columns(2)
    with extra_col_a:
        st.markdown(
            f'<div class="info-box">Highest risk score: '
            f'{"n/a" if np.isnan(max_risk) else f"{max_risk:.4f}"}</div>',
            unsafe_allow_html=True,
        )
    with extra_col_b:
        st.markdown(
            f'<div class="info-box">Successful rows: {quality["success"]} &nbsp; | &nbsp; '
            f'Failed rows: {quality["failed"]}</div>',
            unsafe_allow_html=True,
        )

    render_batch_charts(result_df)

    table_df = result_df.copy()
    for column in ["risk_score", "threshold", "decision_margin"]:
        if column in table_df:
            table_df[column] = pd.to_numeric(table_df[column], errors="coerce").round(4)
    render_table(table_df, max_height=430)

    st.download_button(
        "Download Prediction Results",
        data=st.session_state.get("download_csv_bytes", result_df.to_csv(index=False).encode("utf-8")),
        file_name="machine_prediction_results.csv",
        mime="text/csv",
        width="stretch",
        key=f"download_prediction_results_{st.session_state.get('result_mode') or 'batch'}",
        on_click="ignore",
    )


def render_result_area() -> None:
    section_rule()
    section_heading("Prediction Result")

    if st.session_state.result_mode == "manual" and st.session_state.latest_prediction is not None:
        render_manual_result(st.session_state.latest_prediction)
    elif st.session_state.batch_predictions is not None:
        render_batch_result(st.session_state.batch_predictions)
    else:
        st.markdown(
            """
            <div class="empty-state">
                No prediction result yet. Enter machine data first to view the analysis.
            </div>
            """,
            unsafe_allow_html=True,
        )


def detail_options(result_df: pd.DataFrame, details: dict[int, dict[str, Any]]) -> list[tuple[int, str]]:
    options = []
    for index in details.keys():
        if index >= len(result_df):
            continue
        row = result_df.iloc[index]
        machine_id = str(row.get("Machine ID", "")).strip()
        if machine_id:
            label = f"{machine_id} - row {index + 1}"
        else:
            label = f"Row {index + 1}"
        options.append((index, label))
    return options


def render_explanation_detail(prediction_detail: dict[str, Any]) -> None:
    st.markdown(
        f'<div class="info-box">{build_prediction_narrative(prediction_detail)}</div>',
        unsafe_allow_html=True,
    )

    input_col, scaled_col = st.columns(2)
    with input_col:
        st.caption("Original input")
        original_table = (
            pd.DataFrame([prediction_detail.get("original_input", {})])
            .T.reset_index()
            .rename(columns={"index": "Feature", 0: "Value"})
        )
        render_table(original_table, max_height=280)
    with scaled_col:
        st.caption("Scaled input")
        scaled_df = (
            pd.DataFrame([prediction_detail.get("scaled_input", {})])
            .T.reset_index()
            .rename(columns={"index": "Feature", 0: "Value"})
        )
        scaled_df["Value"] = pd.to_numeric(scaled_df["Value"], errors="coerce").round(4)
        render_table(scaled_df, max_height=280)

    membership_table = format_membership_table(prediction_detail)
    st.caption("Fuzzy membership degree")
    render_table(membership_table, max_height=320)

    membership_long = membership_long_format(prediction_detail)
    if not membership_long.empty:
        fig_membership = px.bar(
            membership_long,
            x="Feature",
            y="Degree",
            color="Membership",
            barmode="group",
            color_discrete_map={"Low": "#94A3B8", "Medium": "#2563EB", "High": "#111827"},
        )
        fig_membership.update_layout(
            title="Membership Degree by Feature",
            xaxis_title="Feature",
            yaxis_title="Membership Degree",
            margin=dict(l=10, r=10, t=42, b=80),
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
        )
        st.plotly_chart(fig_membership, config=PLOTLY_CONFIG)

    contributions = contribution_table(prediction_detail)
    if contributions is None:
        note = contribution_note(prediction_detail)
        if note:
            st.info(note)
    else:
        contributions_display = contributions.copy()
        contributions_display["Contribution"] = contributions_display["Contribution"].round(4)
        contributions_display["Impact"] = contributions_display["Impact"].round(4)
        st.caption("Feature contribution")
        render_table(contributions_display, max_height=280)

        fig_contribution = px.bar(
            contributions.sort_values("Impact", ascending=True),
            x="Contribution",
            y="Feature",
            orientation="h",
            color="Contribution",
            color_continuous_scale=["#166534", "#F3F4F6", "#991B1B"],
        )
        fig_contribution.update_layout(
            title="Feature Contribution to the Model Linear Output",
            xaxis_title="Contribution",
            yaxis_title="Feature",
            margin=dict(l=10, r=10, t=42, b=10),
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_contribution, config=PLOTLY_CONFIG)


def render_explanation_area() -> None:
    details = st.session_state.get("prediction_details", {})
    if not details:
        return

    with st.expander("Why did this result appear?", expanded=True):
        selected_detail = None

        if st.session_state.result_mode == "manual":
            selected_detail = st.session_state.latest_prediction
        else:
            result_df = st.session_state.batch_predictions
            options = detail_options(result_df, details) if result_df is not None else []
            if options:
                label_by_index = dict(options)
                indices = list(label_by_index.keys())
                if st.session_state.selected_prediction not in indices:
                    st.session_state.selected_prediction = indices[0]
                selected_index = st.selectbox(
                    "Select a record to explain",
                    options=indices,
                    format_func=lambda index: label_by_index.get(index, f"Row {index + 1}"),
                    key="selected_prediction",
                )
                selected_detail = details.get(selected_index)

        if selected_detail:
            render_explanation_detail(selected_detail)
        else:
            st.info("No successful row is available for explanation yet.")


def render_system_info(artifacts) -> None:
    metrics = artifacts.model.get("metrics", {})
    threshold = float(artifacts.model["threshold"])
    model_accuracy = metrics.get("Accuracy")
    model_f1 = metrics.get("F1-score")

    with st.expander("About the model and system use", expanded=False):
        st.markdown(
            f"""
            <div class="system-grid">
                <div class="system-item"><strong>Main model</strong><br>ANFIS + Genetic Algorithm.</div>
                <div class="system-item"><strong>Model input</strong><br>5 machine operating parameters.</div>
                <div class="system-item"><strong>Model output</strong><br>Risk score and Normal/Failure Risk status.</div>
                <div class="system-item"><strong>Threshold</strong><br>Loaded from the model: {threshold:.4f}.</div>
                <div class="system-item"><strong>GA optimization</strong><br>GA optimizes membership function parameters and threshold based on the notebook workflow.</div>
                <div class="system-item"><strong>Stored performance</strong><br>Accuracy: {model_accuracy if model_accuracy is not None else "n/a"}; F1-score: {model_f1 if model_f1 is not None else "n/a"}.</div>
                <div class="system-item"><strong>Decision validation</strong><br>The system is a prediction aid, not the final decision maker.</div>
                <div class="system-item"><strong>Data privacy</strong><br>The app does not permanently store uploaded or generated data.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="system-note">The system is designed to support maintenance decisions, not to replace human judgment. Any recommendation produced by the model should be reviewed and validated by a technician or machine operator before maintenance actions are taken. Operational machine data may also contain sensitive industrial information, especially when it comes from a real production site, such as machine performance, operating conditions, location, or production patterns. Therefore, uploaded data should be handled carefully and used only for prediction and analysis purposes.</p>',
            unsafe_allow_html=True,
        )


def main() -> None:
    init_session_state()

    try:
        artifacts = get_artifacts()
        render_header()
    except ArtifactError as exc:
        render_header()
        st.error(str(exc))
        st.stop()

    try:
        stats = get_dataset_stats(tuple(artifacts.feature_columns))
    except Exception as exc:
        st.error(f"Failed to read dataset statistics for validation and simulation: {exc}")
        st.stop()

    render_input_area(artifacts, stats)
    render_result_area()
    render_explanation_area()
    render_system_info(artifacts)

    st.markdown(
        """
        <div class="app-footer">
            <div>Early Industrial Machine Failure Prediction System using ANFIS and Genetic Algorithm</div>
            <div class="app-footer-brand">© Soft Computing 2026</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
