Analysis of Fatalities in Road Accidents Due to Ignorance of Helmets and Seat-Belts
A Case Study

A data-driven analysis and forecasting project that examines road accident fatalities in India caused by non-compliance with helmet and seatbelt safety measures, using exploratory data analysis, time-series models, and ensemble learning.

🚀 Overview

Road traffic accidents remain a major public safety concern in India, with a significant number of fatalities linked to the non-usage of helmets and seat belts.
This project analyzes officially published Indian road accident data from 2017–2022 to:

Understand fatality trends across states and years

Compare helmet vs seatbelt non-compliance impact

Evaluate classical time-series models under real-world data constraints

Improve robustness through ensemble modeling

The work is supported by a published research paper and emphasizes interpretability, diagnostics, and realistic model evaluation.

🧠 Methodology

Data collection from MoRTH and official government sources

Data cleaning and preprocessing

Exploratory Data Analysis (EDA) at national and state levels

Time-series modeling and diagnostics

Machine learning–based forecasting

Ensemble model construction and comparison

Evaluation using RMSE, MAPE, AIC, and BIC

🛠️ Tech Stack

Language: Python

Libraries: Pandas, NumPy, Matplotlib, Seaborn

Modeling: Statsmodels, Scikit-learn

Environment: Jupyter Notebook

📊 Exploratory Data Analysis (EDA)

The EDA focuses on identifying temporal trends, state-wise disparities, and proportional impact of helmet and seatbelt non-compliance.

Included EDA Visualizations

Total fatalities by year (Helmet vs Seatbelt)

State-wise heatmaps (Drivers & Passengers, 2017–2022)

State-wise fatality comparison for 2022

Long-term trend lines for drivers and passengers

Proportional contribution (pie charts) for latest year

These plots highlight regional concentration, temporal variability, and the dominance of helmet-related fatalities.

## 📊 Exploratory Data Analysis (EDA)

### Year-wise Fatality Trends
![Year-wise Trend](results/plots/eda_yearwise_trend.png)

### State-wise Fatalities Heatmap (2017–2022)
![State-wise Heatmap](results/plots/eda_statewise_heatmap.png)

### State-wise Fatalities in 2022
![State-wise 2022](results/plots/eda_statewise_2022.png)

### Proportional Contribution (Latest Year)
![Proportional Contribution](results/plots/eda_proportional_pie.png)


⏱️ Time-Series Analysis

Classical time-series models were applied to evaluate their effectiveness on noisy, sparse accident data.

Models Used

Moving Average (MA)

Auto-Regressive (AR)

ARIMA

Diagnostic Plots Included

Autocorrelation (ACF) and Partial Autocorrelation (PACF)

Trend, seasonality, and residual decomposition

First-order differencing

Actual vs forecasted comparisons

Results show strong volatility, weak stationarity, and sensitivity to outliers, limiting long-horizon forecasting accuracy.

## ⏱️ Time-Series Modeling

### ACF & PACF Analysis
![ACF PACF](results/plots/ts_acf_pacf.png)

### Actual vs Forecasted Fatalities
![Forecast](results/plots/ts_actual_vs_forecast.png)


🤖 Machine Learning & Ensemble Modeling

To improve robustness, multiple models were combined and evaluated.

Models

ARIMA

Gradient Boosting (GBM)

LSTM

Weighted Ensemble

Evaluation & Diagnostics

Residual distribution comparison

Q–Q plots for normality assessment

RMSE comparison across models

MAPE comparison across models

Residual trend analysis for ensemble model

The ensemble approach reduced variance compared to individual models but still reflected inherent data limitations.

## 🤖 Ensemble Modeling & Evaluation

### RMSE Comparison Across Models
![RMSE](results/plots/ensemble_rmse.png)

### Ensemble Model Predictions
![Ensemble](results/plots/ensemble_prediction.png)


📌 Key Findings

Helmet non-compliance contributes significantly more to fatalities than seatbelt non-compliance

Traditional time-series models struggle with real-world accident data variability

High MAPE values indicate poor reliability for long-term forecasting

Ensemble models improve stability but cannot fully overcome data sparsity

Model diagnostics are critical for responsible interpretation

📂 Project Structure
road-accident-fatality-analysis/
│
├── data/            # processed datasets
├── notebooks/       # EDA, time-series, ensemble modeling
├── src/             # reusable preprocessing & modeling code
├── results/         # plots, figures, evaluation outputs
├── paper/           # published research paper (PDF)
└── README.md

📄 Research Paper

This project is based on the paper:

“Analysis of Fatalities in Road Accidents Due to Ignorance of Helmets and Seat-Belts: A Case Study”

📄 Available in the paper/ directory.

🎓 What I Learned

Real-world time-series data is highly noisy and non-stationary

Model accuracy alone is insufficient without diagnostics

Classical models have limited applicability to policy-scale forecasting

Ensemble learning improves robustness but is not a silver bullet

🔮 Future Scope

Incorporating richer temporal and regional features

Improved deep learning architectures with external covariates

Interactive dashboard for policy and safety analysis

Integration of post-2022 accident datasets
