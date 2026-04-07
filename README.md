# 🛡️ Supply Chain Disruption Radar (2026)
**A Real-Time Predictive Intelligence Engine for Global Logistics**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge.svg)](http://localhost:8501/) 
![Accuracy](https://img.shields.io/badge/Accuracy-83.4%25-green)
![Python](https://img.shields.io/badge/Python-3.12-blue)

## 📖 Overview
In the volatility of 2026, waiting for a disruption is not an option. This project is a production-grade MLOps pipeline that monitors geopolitical news, commodity prices, and risk indices to predict supply chain failures before they happen.

The system uses an **Ensemble ML approach** (XGBoost + LightGBM) combined with **NLP sentiment analysis** to generate actionable risk scores.

---

## 🏗️ The 7-Phase Architecture
This project follows a modular design pattern:

1. **Ingestion Layer:** Real-time data fetching from `NewsData.io`, `Yahoo Finance`, and `Geo Risk` indices.
2. **Processing Pipeline:** Automated cleaning and merging of unstructured text and structured time-series data.
3. **Cognitive NLP:** Leveraging **FinBERT** to quantify sentiment from global headlines.
4. **The Intelligence Brain:** An ensemble model combining **XGBoost** and **LightGBM** for high-precision risk classification.
5. **Forecasting Engine:** Utilizing **Meta Prophet** for 28-day predictive trend analysis.
6. **API Layer:** A fully containerized **FastAPI** backend for external system integration.
7. **Executive Dashboard:** A live **Streamlit** interface for real-time monitoring and "War Room" visualization.

---

## 📊 Live Insights (As of April 7, 2026)
The model is currently tracking significant volatility in key commodities:
* **Crude Oil (CL=F):** Trading at **$111.23** 🛢️
* **Aluminum (ALI=F):** Trading at **$3405.00** 🏗️
* **Copper (HG=F):** Trending up **+1.87%** at **$5.67** ⚡

---

## 🛠️ Tech Stack
* **Language:** Python 3.12
* **Machine Learning:** XGBoost, LightGBM, Scikit-learn
* **Time Series:** Prophet (Meta)
* **NLP:** HuggingFace (FinBERT)
* **Data Ops:** FastAPI, Streamlit, Airflow, Docker
* **Architecture Partner:** Optimized with **Claude AI** 🧠

---

## 🚀 Installation & Setup
1. Clone the repository:
   ```bash
   git clone [https://github.com/ayush2309-wq/supply-chain-radar.git](https://github.com/ayush2309-wq/supply-chain-radar.git)
  pip install -r requirements.txt
  streamlit run src/dashboard/app.py
  







   
