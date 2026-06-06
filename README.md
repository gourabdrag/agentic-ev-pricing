# OP'26 Analytics — Agentic AI-Based Dynamic Tariff Optimization for EV Charging Networks

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)]()
[![License](https://img.shields.io/badge/License-Academic-green.svg)]()
[![Status](https://img.shields.io/badge/Status-Complete-success.svg)]()

> **Society of Business — Open Project 2026**  
> *Designing a self-improving pricing engine for EV charging infrastructure*

---

## 📋 Problem Statement

The rapid electrification of mobility has exposed a critical gap in EV charging infrastructure — **static, fixed-rate tariff models** that remain blind to real-world operational dynamics. As EV adoption accelerates, charging stations operating on flat ₹/kWh pricing face:

- **Peak-hour congestion** and long wait times
- **Charger underutilization** during off-peak windows
- **Rising electricity procurement costs**
- **Deteriorating user experience**

This project builds an **Agentic AI framework** — a self-improving pricing engine that autonomously predicts demand, recommends dynamic tariffs in real time, and continuously learns from outcomes.

---

## 🎯 Objectives

| Objective | Description |
|-----------|-------------|
| **Demand Forecast** | Predict charging demand and station utilization across time of day, day of week, and location |
| **Dynamic Tariff** | Optimize per-kWh tariff to maximize revenue while minimizing congestion |
| **Charger Utilization** | Identify underutilized/overloaded stations and optimal time windows |
| **Congestion Reduction** | Smooth demand distribution across time slots to reduce peak-hour queues |
| **Autonomous Intelligence** | Self-improving system with continuous feedback loop |

---

## 📊 Datasets

### 1. ACN-Data (Adaptive Charging Network)
- **Source**: [Caltech EV Dataset](https://ev.caltech.edu/dataset.html)
- **Coverage**: 16,304 EV charging sessions from Caltech and JPL sites
- **Period**: April 25, 2018 — December 16, 2018
- **Key Fields**: Session timestamps, energy delivered (kWh), station IDs, user behavior

### 2. UrbanEV Dataset (ST-EVCDP — Shenzhen)
- **Source**: [IntelligentSystemsLab/ST-EVCDP](https://github.com/IntelligentSystemsLab/ST-EVCDP)
- **Coverage**: 247 grid zones, 1,706 charging stations, 5-minute interval data
- **Period**: June–July 2022 (8,640 time steps)
- **Key Fields**: Occupancy, price, duration, volume, spatial adjacency, station metadata

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    AGENTIC AI FRAMEWORK                          │
│                                                                  │
│  ┌─────────────────┐   ┌─────────────────┐   ┌───────────────┐  │
│  │   DEMAND         │   │   TARIFF         │   │  MONITORING   │  │
│  │   PREDICTION     │──▶│   PRICING        │──▶│  & LEARNING   │  │
│  │   AGENT          │   │   AGENT          │   │  AGENT        │  │
│  │                  │   │                  │   │               │  │
│  │  • XGBoost       │   │  • Surge >80%    │   │  • Feedback   │  │
│  │  • Random Forest │   │  • Discount <30% │   │    Loop       │  │
│  │  • Lag Features  │   │  • Elasticity    │   │  • Parameter  │  │
│  │  • Temporal      │   │    Modeling      │   │    Tuning     │  │
│  └─────────────────┘   └─────────────────┘   └───────────────┘  │
│          │                      │                      │         │
│          ▼                      ▼                      ▼         │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              UNIFIED DATA LAYER (Preprocessed)              │ │
│  │   ACN Sessions (16K+)  |  UrbanEV Grid Data (247 zones)    │ │
│  └─────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- pip package manager

### Installation

```bash
# Clone or navigate to the project directory
cd socbiznew

# Install dependencies
pip install -r requirements.txt
```

### Run the Pipeline

```bash
# Full pipeline (all 5 phases)
python main.py

# EDA only (faster, for exploration)
python main.py --eda-only
```

### View Results

```bash
# Open the interactive dashboard
# Navigate to dashboard/index.html in your browser

# Or view generated plots in outputs/figures/
# And metrics in outputs/metrics/
```

---

## 📁 Project Structure

```
socbiznew/
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
├── main.py                                # End-to-end pipeline runner
│
├── src/                                   # Source code modules
│   ├── __init__.py
│   ├── utils.py                           # Shared utilities & constants
│   ├── data_preprocessing.py              # Data loading & feature engineering
│   ├── eda.py                             # Exploratory Data Analysis
│   ├── demand_prediction_agent.py         # Agent 1: Demand forecasting
│   ├── tariff_pricing_agent.py            # Agent 2: Dynamic pricing
│   └── monitoring_agent.py                # Agent 3: Feedback & learning
│
├── outputs/
│   ├── figures/                           # All generated visualizations (29+ plots)
│   └── metrics/                           # Evaluation metrics & results CSVs
│
├── dashboard/
│   ├── index.html                         # Interactive results dashboard
│   ├── style.css                          # Premium dark-mode styling
│   └── app.js                             # Dashboard interactivity
│
├── presentation/
│   └── EV_Tariff_Optimization_Presentation.pptx (8 slides)
│
└── Datasets OP'26 Analytics-.../          # Raw datasets (ACN + UrbanEV)
```

---

## 📈 Evaluation Metrics

### Demand Prediction Agent
| Metric | Description |
|--------|-------------|
| **RMSE** | Penalizes large errors in predicted station utilization |
| **MAE** | Average absolute error in predicted demand across time slots |
| **R² Score** | How well the model explains variance in actual charging demand |

### Tariff Pricing Agent
| Metric | Description |
|--------|-------------|
| **Revenue Gain %** | ((New − Old) / Old) × 100 vs ₹15/kWh baseline |
| **Charger Utilization Rate** | Before/after dynamic pricing comparison |
| **Off-Peak Uplift** | Session increase in low-demand periods after discounts |

### Monitoring & Learning Agent
| Metric | Description |
|--------|-------------|
| **Avg Wait Time Reduction** | Queue length reduction across peak periods |
| **Customer Response Rate** | Session volume shift in response to tariff changes |
| **Pricing Efficiency Score** | Revenue per kWh delivered over time |

---

## 🔧 Methodology

### Data Preprocessing
- Timestamp normalization and alignment across datasets
- Feature engineering: utilization rate, revenue per session, queue length proxy
- Transparent missing value handling with documented assumptions
- Temporal features: hour, day_of_week, period classification (peak/shoulder/off-peak)

### Demand Prediction
- **Models**: Random Forest and XGBoost with lag features and temporal encodings
- **ACN**: Predicts per-session charger utilization rate
- **UrbanEV**: Predicts grid-level occupancy rate with spatial context

### Dynamic Tariff Optimization
- Surge pricing (1.3x–1.8x) when utilization exceeds 80%
- Discount pricing (0.6x–0.85x) when utilization falls below 30%
- Demand elasticity modeling (price elasticity ≈ -0.3)
- Revenue maximization with congestion minimization

### Monitoring & Learning
- Iterative feedback loop across 10 evaluation episodes
- Parameter refinement based on revenue and utilization outcomes
- Convergence tracking and learning curve visualization

---

## ⚠️ Limitations & Assumptions

- **No causal claims** are made unless clearly justified by the data
- Revenue projections assume a linear demand elasticity model (ε ≈ -0.3)
- The ₹15/kWh baseline is used as a reference; actual tariffs may vary
- ACN data is from US workplace settings; UrbanEV from Chinese urban environments — cross-dataset comparisons account for contextual differences
- Sessions exceeding 48 hours are treated as data errors and removed
- The monitoring agent uses simulated feedback rather than live operational data

---

## 📑 Deliverables Summary

| Deliverable | Status | Location |
|-------------|--------|----------|
| Clean, reproducible code | ✅ | `src/` directory |
| Preprocessing pipeline | ✅ | `src/data_preprocessing.py` |
| EDA visualizations | ✅ | `outputs/figures/01-14_*.png` |
| Demand prediction models & results | ✅ | `outputs/figures/15-19_*.png` + metrics CSV |
| Dynamic tariff optimization | ✅ | `outputs/figures/20-24_*.png` + metrics CSV |
| Monitoring & feedback loop | ✅ | `outputs/figures/25-29_*.png` + metrics CSV |
| Interactive dashboard | ✅ | `dashboard/index.html` |
| Presentation Slide Deck | ✅ | `presentation/EV_Tariff_Optimization_Presentation.pptx` |
| Supporting documentation | ✅ | This README |

---



---

*Built with Python, scikit-learn, XGBoost, matplotlib, and seaborn.*
