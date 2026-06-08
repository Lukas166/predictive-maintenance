# Predictive Maintenance

Streamlit application for early industrial machine failure-risk prediction using an ANFIS + Genetic Algorithm model.

## Run the Application

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Features

- Manual input for 5 machine operating parameters.
- CSV upload for batch prediction.
- Simulation data generation based on `dataset/ai4i2020.csv` statistics.
- Prediction result with risk score, model threshold, status, decision margin, and recommendation.
- Explanation panel with fuzzy membership and feature contribution when the model structure supports safe interpretation.

The model, scaler, notebook, and original dataset are not modified by this app. The app is used only for inference and result visualization.
