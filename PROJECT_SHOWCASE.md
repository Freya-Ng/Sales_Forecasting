# Project Showcase: Sales Forecasting with Explainable AI

## Motivation

Retail businesses need accurate demand forecasting to optimize inventory, staffing, and promotions. However, most ML forecasting models are black boxes — they produce numbers but don't explain *why*, making it hard for business teams to trust or act on them.

This project addresses both problems: it builds a high-accuracy forecasting model **and** makes it transparent using explainability techniques, so business stakeholders can understand the drivers behind each prediction and make informed decisions.

## Approach

The project follows a structured pipeline:

1. **Data Generation** — Generate synthetic sales and weather data for 10 stores, 35 products, and 2 years using `src/data_generator/data_generator.py`.
2. **Data Cleaning** — Handle missing values and outliers. Merge sales with weather data.
3. **Exploratory Analysis** — Identify temporal patterns, category performance, and weather correlations.
4. **Feature Engineering** — Create 55 predictive features from raw data.
5. **Model Training & Tuning** — Train baseline (Prophet) and primary model (LightGBM), optimize with Bayesian search.
6. **Explainability** — Apply SHAP to explain predictions at global and individual levels.
7. **Deployment** — Serve everything through an interactive Streamlit web app with business recommendations.

**[Live Demo](https://saleforecasting.streamlit.app/)**

---

## Techniques & Methods

### Data Preprocessing

| Technique | Purpose |
|-----------|---------|
| Mean imputation | Fill missing sales values (~1% of data) |
| Z-score outlier detection | Identify and cap extreme values (threshold: 3 std) |
| Multi-source integration | Merge sales + weather data by date and province |

### Feature Engineering (55 features)

| Category | Features | Method |
|----------|----------|--------|
| Lag features | `sales_lag_1`, `lag_7`, `lag_14`, `lag_28` | Shifted time series values |
| Rolling statistics | Mean, min, max, std over 7/14/28-day windows | `pandas.rolling()` |
| EWMA | 7/14/28-day spans at alpha 0.5 and 0.75 | Exponentially weighted moving average |
| Store/item aggregations | `store_mean_7d`, `store_sum_7d`, `item_mean_7d`, `item_sum_7d` | Group-level rolling features |
| Temporal | Year, month, day, day of week, quarter, is_weekend, is_holiday | Date decomposition |
| Weather | Temperature, humidity (continuous + binned), season | One-hot encoding + binning |

### Models

| Model | Type | Role |
|-------|------|------|
| **Facebook Prophet** | Additive/multiplicative time series | Baseline model with weekly + yearly seasonality and external regressors |
| **LightGBM** | Gradient boosted decision trees | Primary model trained on 55 engineered features |

### Hyperparameter Tuning

| Aspect | Detail |
|--------|--------|
| Framework | **Optuna** (Bayesian optimization) |
| Parameters tuned | `num_leaves`, `learning_rate`, `feature_fraction`, `bagging_fraction`, `bagging_freq`, `min_child_samples` |
| Trials | 20 |
| Early stopping | 50 rounds |
| Objective | Minimize RMSE |

### Validation Strategy

| Aspect | Detail |
|--------|--------|
| Train/test split | Temporal split at October 2017 (no data leakage) |
| Cross-validation | `TimeSeriesSplit` with 5 folds |
| Holdout test set | October - December 2017 |

### Evaluation Metrics

| Metric | Formula | Why |
|--------|---------|-----|
| **MAE** | Mean Absolute Error | Interpretable error magnitude |
| **RMSE** | Root Mean Squared Error | Penalizes large errors |
| **WAPE** | Weighted Absolute Percentage Error | Business-friendly percentage metric |

### Results

| Model | MAE | RMSE | WAPE |
|-------|-----|------|------|
| Prophet | 8.90 | 11.49 | 29.14% |
| LightGBM (base) | 7.71 | 11.88 | 25.26% |
| LightGBM (tuned) | 7.64 | 11.76 | 24.82% |

### Explainability (XAI)

| Technique | Scope | Purpose |
|-----------|-------|---------|
| **SHAP TreeExplainer** | Global | Rank feature importance across all predictions |
| **SHAP beeswarm plots** | Global | Show how feature values affect predictions |
| **SHAP dependency plots** | Global | Visualize individual feature-outcome relationships |
| **SHAP force plots** | Local | Explain individual predictions (why this number?) |
| **Temporal SHAP analysis** | Global | Reveal monthly and day-of-week prediction patterns |

**Key finding:** Item-level features (44.2%) and sales history (32.5%) dominate predictions. Weather contributes < 1%.

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Language | Python |
| ML models | LightGBM, Prophet |
| Optimization | Optuna |
| Explainability | SHAP |
| Data processing | pandas, NumPy, scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Web app | Streamlit |
| Data formats | CSV, Feather, Pickle, JSON |

---

## Streamlit App

The interactive web app provides three pages accessible via sidebar:

- **Historical Sales Analysis** — Filterable dashboard with KPIs, time series trends, day-of-week patterns, category/store breakdowns, and auto-generated business insights
- **Business Insights & Recommendations** — Store/product performance rankings, seasonal analysis, store-category gap heatmap, growth opportunities, and prioritized revenue-boosting strategies
- **Sales Prediction** — Predict sales for any store-product-date with adjustable weather, promotion, competition, and supply chain factors, plus plain-language interpretation of results

**[Live Demo](https://saleforecasting.streamlit.app/)**
