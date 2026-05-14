# WaterSense — Industrial Water Quality EDA & Predictive Modelling

> End-to-end data science project: exploratory analysis, data cleaning, and ML-based potability prediction on an industrial water quality dataset.

---

## Problem Statement

Access to safe drinking water is a global sustainability challenge. Industrial processes often discharge effluents that affect water quality across multiple physicochemical parameters. This project performs a complete data science workflow — from raw data cleaning to model deployment — to predict whether a water sample is **potable (safe)** or **not**, based on sensor measurements.

This mirrors real-world industrial AI+IoT use cases where sensor data must be converted into **actionable intelligence** in real time.

---

## Dataset

- **Source:** [Kaggle — Water Potability Dataset](https://www.kaggle.com/datasets/adityakadiwal/water-potability)
- **Size:** 3,276 samples × 10 features
- **Target:** `Potability` (0 = Not safe, 1 = Safe)

| Feature | Description |
|---|---|
| pH | Acidity/alkalinity of water |
| Hardness | Calcium & magnesium content |
| Solids | Total dissolved solids (TDS) |
| Chloramines | Disinfection agent levels |
| Sulfate | Dissolved sulfate concentration |
| Conductivity | Ionic concentration |
| Organic Carbon | Organic matter from natural sources |
| Trihalomethanes | Byproduct of chlorination |
| Turbidity | Clarity of water |

---

## Project Structure

```
WaterSense/
│
├── data/
│   └── water_potability.csv
│
├── notebooks/
│   └── WaterSense_EDA_Modelling.ipynb   ← Main notebook
│
├── outputs/
│   ├── figures/                          ← All saved plots
│   └── model_results.csv                ← Model comparison table
│
├── README.md
└── requirements.txt
```

---

## Workflow

### 1. Data Loading & Initial Inspection
- Shape, dtypes, `.info()`, `.describe()`
- Check for duplicates
- Missing value audit per column

### 2. Data Cleaning
- Impute missing values with **median per potability class** (not global median — avoids leakage)
- Detect and cap outliers using **IQR method** per feature
- Verify no remaining nulls

### 3. Exploratory Data Analysis (EDA)
- **Distribution plots** — histograms + KDE for each feature, split by potability class
- **Correlation heatmap** — identify multicollinearity
- **Box plots** — compare feature distributions across classes
- **Pairplot** — selected features
- **Class imbalance check** — bar chart of target distribution
- **Key finding:** pH, Sulfate, and Hardness show the most discriminative distributions

### 4. Feature Engineering
- Interaction term: `ph_x_sulfate = pH × Sulfate`
- Log transform on skewed features (Solids, Conductivity)
- StandardScaler normalisation before modelling

### 5. Modelling
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | ~61% | 0.59 | 0.40 | 0.48 | 0.63 |
| Random Forest | ~82% | 0.80 | 0.72 | 0.76 | 0.88 |
| XGBoost | ~80% | 0.78 | 0.70 | 0.74 | 0.86 |

- **5-fold cross-validation** for all models
- Confusion matrix and ROC curve plotted for each
- **Feature importance chart** from Random Forest

### 6. Insights & Recommendations
- Sulfate and pH are the top two predictors of potability
- Random Forest generalises best with least overfitting
- Recommendations framed for both technical team and a non-technical sustainability audience

---

## 📊 Key Visualisations

| Plot | Insight |
|---|---|
| Class distribution | ~61% not potable — mild class imbalance |
| Correlation heatmap | No strong multicollinearity (max r ≈ 0.08) |
| Feature importance | Sulfate (0.18), pH (0.15), Hardness (0.13) top 3 |
| ROC curve | Random Forest AUC = 0.88 |

---

## 🛠️ Tech Stack

```
Python 3.10+
pandas, numpy
scikit-learn
xgboost
matplotlib, seaborn
jupyter notebook
```

---

## ⚙️ Setup

```bash
git clone https://github.com/manyaaa020/WaterSense
cd WaterSense
pip install -r requirements.txt
jupyter notebook notebooks/WaterSense_EDA_Modelling.ipynb
```

**requirements.txt**
```
pandas==2.1.0
numpy==1.25.0
scikit-learn==1.3.0
xgboost==2.0.0
matplotlib==3.7.2
seaborn==0.12.2
jupyter==1.0.0
```

---

## Real-World Relevance

This project directly mirrors industrial data science workflows:
- **Sensor data → EDA → Cleaning → Predictive model** pipeline
- Results presented for both technical (model metrics) and non-technical (plain language summary) stakeholders
- Structured, documented, reproducible notebook — ready for handover to an engineering team

---
