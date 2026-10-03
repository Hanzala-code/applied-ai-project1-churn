# Customer Churn Prediction
## Week 1: Exploratory Data Analysis
### Dataset
- Source: Telco Customer Churn (Kaggle)
- Size: 7,043 customers, 21 features
- Target: Predict customer churn (Yes/No)
### Key Findings 
- Dataset Scale & Target Distribution: The dataset contains 7,043 customers across 21 features. The overall churn rate is approximately 26.5%, showing class imbalance where non-churned customers form the majority.
- Missing Values: The TotalCharges column contained blank spaces/missing values, which were successfully converted to numeric types for clean evaluation.
- Contract Type: Month-to-month contracts have a significantly higher churn rate compared to one-year or two-year contracts, where churn drops drastically.
- Tenure: Newer customers (low tenure, typically under 12 months) are at the highest risk of leaving[cite: 1]. Churn probability declines as customer duration increases.
### Setup
Open the Kaggle notebook or run locally: pip install pandas numpy matplotlib seaborn 


# ML-Powered Customer Churn Analytics
## Week 2:
Predicting which telecom customers will churn using classical ML models
trained on the IBM Telco Customer Churn dataset (7,043 customers, 26.5% churn rate).

## What this project covers
- Leak-free preprocessing pipeline with stratified train/test split
- Baseline → Logistic Regression → Decision Tree → Random Forest
- Confusion matrix, precision, recall, F1 computed by hand and verified with sklearn
- ROC-AUC curve and cost-based threshold selection (PKR 6,000 vs PKR 1,000)
- Odds ratio interpretation of LR coefficients
- Overfitting analysis across Decision Tree depths
- Class imbalance handling with class_weight='balanced'
- Feature engineering evaluated against baseline AUC
- Baseline (always "stay"): accuracy 0.735
- Best model: Random Forest, AUC 0.842, recall 0.856 at threshold 0.20
- Top churn drivers (permutation importance): tenure, TotalCharges, Contract_Two year
- Threshold chosen: 0.20, because a missed churner costs PKR 6,000 in lost revenue vs PKR 1,000 for an unnecessary retention offer — the 6:1 cost ratio makes aggressive flagging rational (t* = 0.14 by decision theory)
- Engineered features: n_services, is_new, charge_per_mo, price_jump; effect on AUC: 0.8422 → 0.8420 (no improvement — Random Forest already captures these interactions internally)
- Biggest lesson: accuracy is a misleading metric on imbalanced data — a model that catches zero churners can still score 73.5%, so always check recall and AUC before trusting any headline number


## Key results
| Model | AUC | Recall (t=0.20) |
|---|---|---|
| Baseline (always Stay) | 0.500 | 0.000 |
| Logistic Regression | 0.840 | — |
| Decision Tree (d=5) | — | — |
| Random Forest | 0.842 | 0.856 |

**Chosen threshold: 0.20** — catches 108 more churners than default 0.5,
saving an estimated PKR 285,000 net over the default threshold.


#  Model Optimization and Unsupervised Learning
## Week 3:

- Split-to-split accuracy range across 20 seeds: 0.780 to 0.828 
  (std 0.0104, theoretical SE 0.0107, 95% CI ±0.021)

- 5-fold CV AUC:
  - LR  (tuned C=10):   0.846 +/- 0.013
  - RF  (random search): 0.844 +/- 0.011
  - XGBoost (tuned):    0.850 +/- 0.012

- Tuning: best RF params — max_depth=15, min_samples_leaf=15, 
  max_features=0.213; random search (24 iterations) completed 
  in 131s and matched grid search quality while exploring 
  a wider parameter space

- XGBoost early stopping chose 247 trees out of 2,000; 
  best params: learning_rate=0.034, max_depth=2, 
  n_estimators=476, reg_lambda=1.974

- Test AUC of final model (XGBoost, used once): 0.8483
  recall 0.521, precision 0.659 — within CV mean ± 2 std ✓

- Customer segments (k=4):
  - "New, High Spenders"  — 2,157 customers, churn 43% → proactive outreach before month 18
  - "New, Budget"         — 1,918 customers, churn 32% → free service bundle for 3 months
  - "Loyal, Heavy Users"  — 1,938 customers, churn 14% → reward programme, upsell
  - "Loyal, Light Users"  — 1,030 customers, churn  5% → minimal intervention needed

- PCA: 15 of 30 components explain 90% of the variance — 
  half the features are redundant, driven by collinear 
  "No internet service" dummy columns (all loading at 0.302 on PC1)

- Biggest lesson: a single train/test split can swing ±0.024 
  accuracy on the same model and data — always report 
  cross-validated mean ± std, never a bare number
