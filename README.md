**# Analysis of Fatalities in Road Accidents Due to Ignorance of Helmets and Seat-Belts**  

**### A Case Study**



**A data-driven analysis and forecasting project that studies road accident fatalities in India caused by non-compliance with helmet and seatbelt safety measures, using time-series and machine learning models.**



**---**



**## 🚀 Overview**



**Road traffic accidents remain a major cause of fatalities in India, with a significant portion linked to the non-usage of helmets and seat belts.**  

**This project analyzes officially published Indian road accident data from \*\*2017–2022\*\* to identify trends, study compliance impact, and forecast future fatality risks.**



**The work is supported by a published research paper and focuses on \*\*interpretable models, rigorous evaluation, and real-world data challenges\*\*.**



**---**



**## 🧠 Methods \& Approach**



**- Data collection from MoRTH and government sources**  

**- Data cleaning, preprocessing, and exploratory analysis**  

**- Time-series forecasting using:**

  **- Moving Average (MA)**

  **- Auto-Regressive (AR)**

  **- ARIMA**

**- Machine learning models for comparative evaluation**  

**- Ensemble modeling to improve forecasting robustness**  

**- Model evaluation using \*\*MAPE, RMSE, AIC, and BIC\*\***



**---**



**## 🛠️ Tech Stack**



**- \*\*Programming Language:\*\* Python**  

**- \*\*Libraries:\*\* Pandas, NumPy, Matplotlib, Seaborn**  

**- \*\*Modeling \& Analysis:\*\* Statsmodels, Scikit-learn**  

**- \*\*Environment:\*\* Jupyter Notebook**  



**---**



**## 📊 Key Findings**



**- Moving Average models captured short-term trends but performed poorly in long-term forecasting**  

**- AR and ARIMA models showed high variance and large forecasting errors due to data sparsity**  

**- Helmet non-compliance trends were slightly more predictable than seatbelt-related fatalities**  

**- High MAPE values highlighted the limitations of traditional models on real-world accident data**  



**---**



**## 📂 Project Structure**



**road-accident-fatality-analysis/**

**│**

**├── data/ # raw and processed datasets**

**├── notebooks/ # data cleaning, EDA, modeling notebooks**

**├── src/ # reusable preprocessing and modeling scripts**

**├── results/ # plots, figures, and evaluation outputs**

**├── paper/ # published research paper (PDF)**

**└── README.md**





**---**



**## 📄 Research Paper**



**This project is based on the research paper:**



**\*\*“Analysis of Fatalities in Road Accidents Due to Ignorance of Helmets and Seat-Belts: A Case Study”\*\***



**The full paper is available in the `paper/` directory.**



**---**



**## 🔍 What I Learned**



**- Real-world time-series data is highly noisy and difficult to model**  

**- Forecasting accuracy is strongly affected by data sparsity and variance**  

**- Model diagnostics are as important as prediction accuracy**  

**- Ensemble approaches can improve stability but are not a silver bullet**  



**---**



**## 🔮 Future Improvements**



**- Integration of deep learning models such as LSTM**  

**- Inclusion of more granular and recent datasets**  

**- Deployment as an interactive dashboard for policy analysis**  



