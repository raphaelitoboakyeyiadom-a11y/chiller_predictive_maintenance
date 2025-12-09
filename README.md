# ❄️ Chiller Predictive Maintenance – Industrial HVAC Dashboard

AI-powered predictive maintenance system for an industrial **water-cooled chiller**.  
It uses synthetic sensor data to detect **efficiency loss** and **critical faults**,  
and estimates the **energy and downtime cost impact**.

---

## 📌 Project Overview

The app simulates sensor inputs from a central plant chiller:

- Evaporator outlet temperature (°C)
- Condenser pressure (bar)
- Chilled water flow rate (m³/h)
- Compressor current (A)
- Refrigerant level (%)
- Cooling tower outlet temperature (°C)

From these, a machine-learning model predicts:

- **Normal** – chiller operating efficiently  
- **Efficiency drop** – fouling, poor tower performance, or partial refrigerant issues  
- **Critical fault** – strong patterns of low flow, high pressure, low refrigerant, or high current  

The dashboard then provides **maintenance recommendations**, plus estimates of:

- Extra energy usage (kWh/month)
- Extra electricity cost (USD/month)
- Avoided downtime hours
- Avoided downtime cost

---

## 🎯 Objectives

- Detect early signs of chiller **efficiency loss and incipient faults**  
- Prioritize **maintenance actions** for technicians  
- Quantify **business impact** in terms of energy and downtime cost  
- Demonstrate a strong **AI + Mechanical Engineering** portfolio project

---

## 🧪 Machine Learning

- Model: **Random Forest Classifier**
- Training data: **synthetic chiller dataset** generated in `app.py`
- Target classes:
  - `Normal`
  - `Efficiency drop`
  - `Critical fault`
- Key outputs:
  - Validation accuracy
  - Confusion matrix
  - Feature importance (which sensors influence the decision most)

---

## 🚀 How to Run

1. **Create and activate a virtual environment**

2. **Install dependencies**

```bash
pip install -r requirements.txt

streamlit run app.py

http://localhost:8501
