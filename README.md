# Industrial Machine Failure Prediction

Early industrial machine failure prediction system using an ANFIS model optimized with a Genetic Algorithm. The application is built with Streamlit and is intended for inference, prediction visualization, simulation, and user-friendly explanation of machine failure risk.

The app does not retrain the model and does not modify the original dataset, notebook, scaler, or model artifacts.

## Project Overview

This project predicts whether an industrial machine is in a normal operating condition or at failure risk based on five operating parameters:

- `Air temperature [K]`
- `Process temperature [K]`
- `Rotational speed [rpm]`
- `Torque [Nm]`
- `Tool wear [min]`

The application loads the trained artifacts from the `models/` directory, applies the same preprocessing and inference flow used in the notebook, and displays:

- Risk score
- Model threshold
- Prediction status
- Decision margin
- Recommended action
- Fuzzy membership explanation
- Batch prediction table
- Simulation results
- Browser tab icon from `public/webIcon.png`

## Project Structure

```text
PREDICTIVE-MAINTENANCE/
|-- app.py
|-- requirements.txt
|-- dataset/
|   `-- ai4i2020.csv
|-- models/
|   |-- anfis_with_ga_params.pkl
|   |-- anfis_without_ga_params.pkl
|   |-- feature_columns.pkl
|   `-- scaler.pkl
|-- notebook/
|   `-- 230011_230057_230079_Kode_Program_UAS_Softcom....ipynb
|-- public/
|   `-- webIcon.png
|-- utils/
|   |-- __init__.py
|   |-- inference.py
|   |-- explanation.py
|   |-- data_generator.py
|   |-- validation.py
|   `-- ui_style.py
`-- README.md
```

## Installation Guide

### 1. Clone or open the project

Open a terminal in the project root directory:

```bash
cd predictive-maintenance
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Running the Application

Run the Streamlit app from the project root:

```bash
streamlit run app.py
```

After the command runs, Streamlit will show a local URL, usually:

```text
http://localhost:8501
```

Open the URL in a browser to use the application.

## How to Use the Application

The application has one main page with two main areas:

- Machine Data Input
- Prediction Result

### Manual Input

Use this mode when you want to predict one machine record.

1. Select `Manual Input`.
2. Enter the machine operating values.
3. Optionally enter machine identity information such as Machine ID.
4. Click `Run Prediction`.
5. Review the risk score, threshold, status, decision margin, recommendation, and explanation panel.

### CSV Upload

Use this mode when you want to predict multiple machine records.

1. Select `CSV / Simulation`.
2. Choose `Upload CSV` as the data source.
3. Download the CSV template if needed.
4. Upload a CSV file with the required columns.
5. Review the preview and validation messages.
6. Choose how missing values should be handled.
7. Click the batch prediction button.
8. Review the result table and download the prediction results if needed.

Required CSV columns:

```text
Air temperature [K]
Process temperature [K]
Rotational speed [rpm]
Torque [Nm]
Tool wear [min]
```

Optional CSV columns:

```text
Machine ID
Timestamp
Location
Operator
```

### Generate Simulation Data

Use this mode when you want to test the application with realistic demo data.

1. Select `CSV / Simulation`.
2. Choose `Generate Simulation` as the data source.
3. Set the number of records.
4. Select a simulation scenario.
5. Generate the simulation data.
6. Download the generated data if needed.
7. Run prediction on the generated records.

Simulation scenarios are based on statistics from `dataset/ai4i2020.csv`, including min, max, mean, standard deviation, median, and quantiles.

## Prediction Output

Each prediction includes:

- `risk_score`: Model output after ANFIS inference and sigmoid transformation.
- `threshold`: Decision threshold loaded from the model artifact.
- `prediction`: Numeric model decision.
- `status`: `Normal` or `Failure Risk`.
- `decision_margin`: Absolute distance between risk score and threshold.
- `recommendation`: User-oriented maintenance recommendation.

Classification logic:

```text
If risk_score >= threshold: Failure Risk
If risk_score < threshold: Normal
```

No additional manual threshold or extra status category is used by the application.

## Model Artifacts

The application uses these files:

- `models/scaler.pkl`
- `models/feature_columns.pkl`
- `models/anfis_with_ga_params.pkl`
- `models/anfis_without_ga_params.pkl`

The main model is `anfis_with_ga_params.pkl`. The threshold is read directly from the model file.

## Notes and Limitations

- This system is a decision-support tool, not a final maintenance decision maker.
- Maintenance decisions must be validated by a technician or operator.
- Operational machine data can be sensitive when it comes from a real industrial site.
- The application does not permanently store uploaded, simulated, or predicted data.
- The application is designed for inference and visualization only, not model training.

## Troubleshooting

If the app fails to start, check that:

- All files in `models/` are available.
- `dataset/ai4i2020.csv` exists.
- Dependencies were installed with `pip install -r requirements.txt`.
- The app is run from the project root using `streamlit run app.py`.

If a CSV upload fails, check that:

- Required columns are present.
- Feature columns contain numeric values.
- Missing values are handled using the provided option.
- Column names match the template exactly.
