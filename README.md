# Sales Forecasting with Explainable AI (XAI)

> End-to-end sales forecasting system combining LightGBM, SHAP explainability, and an interactive Streamlit dashboard with business recommendations.

**Tech Stack:** Python, LightGBM, Prophet, SHAP, Optuna, Streamlit

**[Live Demo](https://your-streamlit-app-link.streamlit.app)** | **[Project Showcase](PROJECT_SHOWCASE.md)** | **[SHAP Analysis Report](docs/shap_analysis_summary_report.md)**

---

## Overview

This project builds a complete sales forecasting pipeline — from raw data to an interactive web app — for 10 retail stores across 2 provinces in Vietnam (Hanoi & Ho Chi Minh City), covering 35 products in 5 categories over 2 years (2016-2017).

The system forecasts daily store-item sales using **LightGBM** (optimized with **Optuna**), explains predictions with **SHAP**, and serves everything through a **Streamlit** web application that provides historical analysis, business insights, and interactive predictions.

## Key Features

- **Data Pipeline:** Preprocessing, missing value imputation, outlier correction, and weather data integration
- **Feature Engineering:** 55 features including lag values, rolling statistics (7/14/28-day), EWMA, temporal features, and weather indicators
- **Baseline Model:** Facebook Prophet with multiplicative seasonality as a time series baseline
- **Primary Model:** LightGBM regressor with TimeSeriesSplit cross-validation (5 folds)
- **Hyperparameter Tuning:** Bayesian optimization via Optuna (6 parameters, 20 trials)
- **Explainability:** SHAP TreeExplainer with global importance, dependency plots, and per-prediction force plots
- **Interactive App:** 3-page Streamlit dashboard with filters, KPIs, business insights, and a prediction tool

## Model Performance

| Model | MAE | RMSE | WAPE |
|-------|-----|------|------|
| Prophet (baseline) | 8.90 | 11.49 | 29.14% |
| LightGBM (base) | 7.71 | 11.88 | 25.26% |
| LightGBM (Optuna-tuned) | 7.64 | 11.76 | 24.82% |

## Streamlit App

The app provides three pages:

1. **Historical Sales Analysis** — filterable dashboard with KPIs, sales trends, category/store breakdowns, and auto-generated quick insights
2. **Business Insights & Recommendations** — performance rankings, seasonal patterns, growth opportunities (store-category heatmap), and prioritized actionable strategies to boost revenue
3. **Sales Prediction** — interactive tool to predict sales for any store-product-date combination with adjustable factors (weather, promotions, competition, supply chain) and business-friendly interpretation

**[Live Demo](https://your-streamlit-app-link.streamlit.app)**

## Project Structure

```
Sales_Forecasting/
├── app.py                                # Streamlit app entry point
├── data/
│   ├── 2016_sales.csv                    # Raw 2016 sales data
│   ├── 2017_sales.csv                    # Raw 2017 sales data
│   ├── weather_data.csv                  # Weather data (temperature, humidity, season)
│   ├── sales_data_preprocessed.csv       # Cleaned sales data
│   ├── weather_preprocessed.csv          # Cleaned weather data
│   └── feature_engineered_data_55_features.feather
├── models/
│   ├── sales_forecast_model.pkl          # Trained LightGBM model
│   └── feature_stats.json               # Feature metadata
├── notebooks/
│   ├── 01_preprocessing.ipynb            # Data cleaning & outlier handling
│   ├── 02_EDA.ipynb                      # Exploratory data analysis
│   ├── 03_feature_engineering.ipynb      # 55-feature creation pipeline
│   ├── 04_modelling.ipynb                # Prophet + LightGBM + Optuna tuning
│   └── 05_explain_model.ipynb            # SHAP explainability analysis
├── src/
│   ├── data_generator/                   # Synthetic data generation
│   ├── data_loader/                      # Data loading with Streamlit caching
│   ├── ui_builder/                       # Dashboard, insights, and visualization modules
│   ├── ui_predictor/                     # Prediction interface with business interpretation
│   └── utils/                            # Plotting and utility functions
├── docs/
│   ├── project_description_poc_phase.md
│   └── shap_analysis_summary_report.md
├── figures/                              # Generated SHAP and EDA visualizations
├── requirements.txt
└── PROJECT_SHOWCASE.md                   # Technical showcase of methods used
```

## Getting Started

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd Sales_Forecasting
```

### 2. Set up the environment

**Option A — Conda:**

```bash
conda env create -f environment.yml
conda activate sales_forecast
```

**Option B — pip:**

```bash
python -m venv .venv

# macOS/Linux:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Generate the dataset

This creates the raw sales and weather CSV files in the `data/` directory.

```bash
python src/data_generator/data_generator.py
```

You can modify parameters in `src/data_generator/data_generator.py` to change the date range, outlier ratio, or missing value ratio.

### 4. Run the notebooks (sequential)

```bash
jupyter lab
```

Run in order: `01_preprocessing` → `02_EDA` → `03_feature_engineering` → `04_modelling` → `05_explain_model`

### 5. Launch the Streamlit app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## How It Works

1. **Data Pipeline** — Sales and weather data are cleaned, merged, and validated. Missing values are imputed with mean; outliers are capped using z-score thresholds.
2. **Feature Engineering** — 55 features are created: lag values (1/7/14/28 days), rolling statistics (mean/min/max/std over 7/14/28 days), EWMA (alpha 0.5 & 0.75), store/item aggregations, temporal features, and one-hot encoded weather indicators.
3. **Model Training** — LightGBM is trained with TimeSeriesSplit cross-validation. Optuna tunes 6 hyperparameters via Bayesian optimization. Prophet serves as the baseline.
4. **Explainability** — SHAP TreeExplainer generates global feature importance rankings, dependency plots, and local force plots for individual predictions.
5. **Web App** — Streamlit serves 3 pages: historical analysis with filters, business insights with actionable recommendations, and an interactive prediction tool.

## References

- [LightGBM](https://lightgbm.readthedocs.io/) — Gradient boosting framework
- [SHAP](https://shap.readthedocs.io/) — Model explainability
- [Optuna](https://optuna.org/) — Hyperparameter optimization
- [Prophet](https://facebook.github.io/prophet/) — Time series forecasting
- [Streamlit](https://streamlit.io/) — Web app framework
