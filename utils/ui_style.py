"""Custom CSS for the Streamlit app."""

from __future__ import annotations

import streamlit as st


def apply_custom_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --app-bg: #F8FAFC;
            --surface: #FFFFFF;
            --border: #E5E7EB;
            --text: #111827;
            --secondary: #6B7280;
            --muted: #9CA3AF;
            --button: #111827;
            --button-hover: #374151;
            --green-bg: #DCFCE7;
            --green-text: #166534;
            --red-bg: #FEE2E2;
            --red-text: #991B1B;
            --amber-bg: #FEF3C7;
            --amber-text: #92400E;
        }

        #MainMenu, footer,
        [data-testid="stHeader"] {
            display: none !important;
        }

        .stApp {
            background: var(--app-bg);
            color: var(--text);
        }

        .block-container,
        div[data-testid="stMainBlockContainer"] {
            max-width: 1180px !important;
            padding-top: 1rem !important;
            padding-bottom: 2.5rem !important;
        }

        .top-app-bar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 16px;
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 12px 16px;
            margin-bottom: 18px;
        }

        .top-app-left {
            display: flex;
            align-items: center;
            gap: 10px;
            min-width: 0;
        }

        .top-app-title {
            color: var(--text);
            font-size: 15px;
            font-weight: 700;
            line-height: 1.3;
            white-space: nowrap;
        }

        .model-badge {
            display: inline-flex;
            align-items: center;
            height: 24px;
            padding: 0 9px;
            border-radius: 6px;
            border: 1px solid var(--border);
            color: #374151;
            background: #F9FAFB;
            font-size: 12px;
            font-weight: 700;
            white-space: nowrap;
        }

        .model-ready, .model-error {
            display: inline-flex;
            align-items: center;
            min-height: 26px;
            padding: 3px 9px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 700;
            white-space: nowrap;
        }

        .model-ready {
            color: var(--green-text);
            background: var(--green-bg);
        }

        .model-error {
            color: var(--red-text);
            background: var(--red-bg);
        }

        .app-header {
            margin: 0 0 24px 0;
        }

        .app-header h1 {
            color: var(--text);
            font-size: 32px;
            font-weight: 760;
            line-height: 1.18;
            letter-spacing: 0;
            margin: 0 0 4px 0;
        }

        .app-header p {
            width: 100%;
            max-width: none;
            color: var(--secondary);
            font-size: 16px;
            line-height: 1.6;
            margin: 0;
        }

        .section-heading {
            margin-bottom: 16px;
        }

        .section-heading h2 {
            color: var(--text);
            font-size: 22px;
            font-weight: 760;
            line-height: 1.3;
            margin: 0 0 6px 0;
        }

        .section-heading p {
            color: var(--secondary);
            font-size: 15px;
            line-height: 1.5;
            margin: 0;
        }

        .section-rule {
            height: 1px;
            background: rgba(17, 24, 39, 0.18);
            margin: 34px 0 22px 0;
            border: 0;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-color: var(--border) !important;
            border-radius: 8px !important;
            background: var(--surface) !important;
            box-shadow: none !important;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] > div {
            padding: 18px 20px !important;
        }

        div[data-testid="stSegmentedControl"] {
            margin-bottom: 16px;
        }

        div[data-testid="stSegmentedControl"] label {
            border-radius: 6px !important;
            border-color: var(--border) !important;
            min-height: 38px !important;
            font-weight: 700 !important;
            font-size: 14px !important;
        }

        div[data-testid="stSegmentedControl"] button[aria-pressed="true"] {
            background: var(--button) !important;
            border-color: var(--button) !important;
            color: #FFFFFF !important;
        }

        div[data-testid="stSegmentedControl"] label:has(input:checked) {
            background: var(--button) !important;
            border-color: var(--button) !important;
            color: #FFFFFF !important;
        }

        div[data-testid="stSegmentedControl"] label:has(input:checked) * {
            color: #FFFFFF !important;
        }

        div[data-testid="stButtonGroup"] button[data-testid="stBaseButton-segmented_controlActive"] {
            background: var(--button) !important;
            border-color: var(--button) !important;
            color: #FFFFFF !important;
        }

        div[data-testid="stButtonGroup"] button[data-testid="stBaseButton-segmented_controlActive"] * {
            color: #FFFFFF !important;
        }

        div[data-testid="stButtonGroup"] button[data-testid="stBaseButton-segmented_control"] {
            background: #FFFFFF !important;
            border-color: var(--border) !important;
            color: #374151 !important;
        }

        div[data-testid="stButtonGroup"] button[data-testid="stBaseButton-segmented_control"]:hover {
            background: #F3F4F6 !important;
            color: #111827 !important;
        }

        div[data-testid="stButtonGroup"] {
            margin-bottom: 16px;
        }

        .stButton,
        .stDownloadButton {
            margin-top: 10px;
        }

        .stButton > button,
        .stDownloadButton > button,
        div[data-testid="stFormSubmitButton"] > button {
            border-radius: 6px !important;
            border: 1px solid var(--button) !important;
            background: var(--button) !important;
            color: #FFFFFF !important;
            font-weight: 700 !important;
            min-height: 40px !important;
            box-shadow: none !important;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover,
        div[data-testid="stFormSubmitButton"] > button:hover {
            border-color: var(--button-hover) !important;
            background: var(--button-hover) !important;
            color: #FFFFFF !important;
        }

        .stNumberInput input,
        .stTextInput input,
        .stSelectbox div,
        .stFileUploader section {
            border-radius: 6px !important;
        }

        .metric-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 10px;
            margin: 6px 0 14px 0;
        }

        .metric-card {
            background: #FFFFFF;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 13px 14px;
            min-height: 88px;
        }

        .metric-label {
            color: var(--secondary);
            font-size: 13px;
            font-weight: 700;
            margin-bottom: 7px;
        }

        .metric-value {
            color: var(--text);
            font-size: 22px;
            font-weight: 780;
            line-height: 1.2;
            overflow-wrap: anywhere;
        }

        .status-badge {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-height: 28px;
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 13px;
            font-weight: 800;
            line-height: 1.2;
        }

        .status-normal {
            background: var(--green-bg);
            color: var(--green-text);
        }

        .status-failure {
            background: var(--red-bg);
            color: var(--red-text);
        }

        .recommendation-box,
        .warning-box,
        .empty-state,
        .info-box {
            border-radius: 8px;
            border: 1px solid var(--border);
            padding: 13px 14px;
            font-size: 15px;
            line-height: 1.55;
        }

        .empty-state {
            margin: 8px 0 22px 0;
        }

        .recommendation-box,
        .info-box {
            margin: 10px 0 14px 0;
        }

        .warning-box {
            margin: 12px 0 14px 0;
        }

        .recommendation-box,
        .info-box,
        .empty-state {
            background: #F9FAFB;
            color: #374151;
        }

        .warning-box {
            background: var(--amber-bg);
            border-color: #FDE68A;
            color: var(--amber-text);
        }

        .muted-note {
            color: var(--secondary);
            font-size: 13px;
            line-height: 1.5;
            margin: 8px 0 14px 0;
        }

        div[data-testid="stDataFrame"] {
            margin-bottom: 6px;
        }

        div[data-testid="stDataFrame"] [role="gridcell"],
        div[data-testid="stDataFrame"] [role="columnheader"] {
            font-size: 14px !important;
        }

        .table-scroll {
            width: 100%;
            overflow-x: auto;
            margin: 8px 0 18px 0;
            border: 1px solid var(--border);
            border-radius: 8px;
            background: #FFFFFF;
        }

        .static-table table {
            width: 100%;
            border-collapse: collapse;
            border-spacing: 0;
            font-size: 14px;
            color: var(--text);
        }

        .static-table th,
        .static-table td {
            border-bottom: 1px solid var(--border);
            border-right: 1px solid var(--border);
            padding: 10px 12px;
            text-align: left;
            vertical-align: middle;
            white-space: nowrap;
        }

        .static-table th:last-child,
        .static-table td:last-child {
            border-right: none;
        }

        .static-table tr:last-child td {
            border-bottom: none;
        }

        .static-table th {
            background: #F9FAFB;
            color: var(--secondary);
            font-weight: 650;
        }

        .static-table td {
            background: #FFFFFF;
        }

        .table-shell {
            border: 1px solid var(--border);
            border-radius: 8px;
            overflow: hidden;
            background: #FFFFFF;
        }

        div[data-testid="stExpander"] {
            border: 1px solid var(--border) !important;
            border-radius: 8px !important;
            background: #FFFFFF !important;
            margin-top: 18px !important;
            margin-bottom: 4px !important;
        }

        div[data-testid="stExpander"] summary {
            font-weight: 750 !important;
            color: var(--text) !important;
            font-size: 15px !important;
        }

        .system-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 10px;
            margin-top: 8px;
        }

        .system-item {
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 12px;
            background: #F9FAFB;
            color: #374151;
            font-size: 14px;
            line-height: 1.45;
        }

        .system-item strong {
            color: var(--text);
        }

        .system-note {
            display: block;
            color: var(--secondary);
            font-size: 13px;
            line-height: 1.5;
            margin: 22px 12px 0 12px;
            padding-top: 14px;
            padding-bottom: 14px;
            border-top: 1px solid rgba(17, 24, 39, 0.12);
        }

        .app-footer {
            border-top: 1px solid rgba(17, 24, 39, 0.18);
            margin-top: 34px;
            padding-top: 16px;
            color: var(--secondary);
            font-size: 13px;
            line-height: 1.6;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 16px;
        }

        .app-footer-brand {
            color: var(--text);
            font-weight: 650;
            text-align: right;
            white-space: nowrap;
        }

        @media (max-width: 760px) {
            .block-container,
            div[data-testid="stMainBlockContainer"] {
                padding-left: 1rem !important;
                padding-right: 1rem !important;
            }

            .top-app-bar,
            .top-app-left {
                align-items: flex-start;
                flex-direction: column;
            }

            .top-app-title {
                white-space: normal;
            }

            .app-header h1 {
                font-size: 26px;
            }

            .metric-grid,
            .system-grid {
                grid-template-columns: 1fr;
            }

            .app-footer {
                align-items: flex-start;
                flex-direction: column;
            }

            .app-footer-brand {
                text-align: left;
                white-space: normal;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
