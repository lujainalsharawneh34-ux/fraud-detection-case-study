# Fraud Detection Case Study

A professional machine learning case study for detecting fraudulent financial transactions using Python.  
The project covers data cleaning, exploratory data analysis, feature engineering, class imbalance handling, model training, model evaluation, and business-focused recommendations.

---

## Project Overview

Fraud detection is a high-impact classification problem where the goal is not only to achieve high accuracy, but also to correctly identify rare fraudulent transactions while keeping false alerts manageable.

This project builds and evaluates machine learning models to classify transactions as:

- `0` — Legitimate transaction
- `1` — Fraudulent transaction

The final recommended model is **Random Forest**, selected because it provides stronger fraud detection performance and a better balance between precision, recall, and PR-AUC compared with the baseline Logistic Regression model.

---

## Repository Structure

```text
fraud-detection-case-study/
│
├── fraud.ipynb          # Main Jupyter Notebook
├── fraud.csv            # Transaction dataset 
└── README.md            # Project documentation
```

> Note: The notebook uses `DATA_PATH = "fraud.csv"`.  
> If your dataset has a different file name, rename it to `fraud.csv` or update the `DATA_PATH` variable in the notebook.

---

## Dataset Summary

The dataset contains **594,643 transactions** and **10 original columns**.

### Original Columns

| Column | Description |
|---|---|
| `step` | Time step of the transaction |
| `customer` | Customer identifier |
| `age` | Customer age group |
| `gender` | Customer gender |
| `zipcodeOri` | Customer origin ZIP code |
| `merchant` | Merchant identifier |
| `zipMerchant` | Merchant ZIP code |
| `category` | Transaction category |
| `amount` | Transaction amount |
| `fraud` | Target variable: `0` legitimate, `1` fraud |

### Class Distribution

| Class | Count | Percentage |
|---|---:|---:|
| Legitimate | 587,443 | 98.79% |
| Fraud | 7,200 | 1.21% |

The dataset is highly imbalanced, with approximately an **82:1 ratio** of legitimate to fraudulent transactions. Because of this imbalance, accuracy alone is not a reliable evaluation metric.

---

## Key Objectives

The main objectives of this case study are to:

1. Understand the dataset and detect data quality issues.
2. Explore fraud patterns across transaction amount, category, merchant, age, gender, and time.
3. Engineer meaningful predictive features.
4. Handle severe class imbalance using SMOTE-NC.
5. Train and compare machine learning models.
6. Evaluate models using fraud-relevant metrics.
7. Recommend a practical model for fraud detection use cases.

---

## Methodology

### 1. Data Preprocessing

The preprocessing stage includes:

- Loading the transaction dataset.
- Removing quote artifacts from categorical columns.
- Checking missing values.
- Reviewing data types.
- Dropping zero-variance columns:
  - `zipcodeOri`
  - `zipMerchant`
- Encoding categorical variables:
  - `age`
  - `gender`
  - `category`

---

### 2. Exploratory Data Analysis

The EDA focuses on understanding fraud behavior across:

- Target class imbalance
- Transaction amount distribution
- Fraud rate by transaction category
- Fraud rate by time patterns
- Fraud rate by age group and gender
- Merchant-level and customer-level patterns

Important insights include:

- Fraudulent transactions are rare but financially significant.
- Transaction amount is a strong signal.
- Merchant and category risk are highly informative.
- Fraud patterns vary by time and transaction category.

---

### 3. Feature Engineering

The notebook creates several feature groups:

#### Temporal Features

| Feature | Description |
|---|---|
| `hour_of_day` | Hour extracted from `step` |
| `day_of_week` | Day extracted from `step` |
| `is_night` | Indicates whether the transaction happened at night |

#### Amount Features

| Feature | Description |
|---|---|
| `amount` | Original transaction amount |
| `log_amount` | Log-transformed transaction amount |

#### Customer Features

| Feature | Description |
|---|---|
| `cust_tx_count` | Number of transactions per customer |
| `cust_avg_amount` | Average transaction amount per customer |
| `cust_max_amount` | Maximum transaction amount per customer |

#### Merchant and Category Features

| Feature | Description |
|---|---|
| `merch_fraud_rate` | Historical fraud rate by merchant |
| `merch_tx_count` | Number of transactions per merchant |
| `cat_fraud_rate` | Historical fraud rate by category |

To avoid target leakage, aggregate features are calculated using the training set only and then mapped to both training and test data.

---

## Class Imbalance Handling

Fraud cases represent only **1.21%** of the dataset, so the notebook applies **SMOTE-NC** only on the training set.

### Before SMOTE-NC

| Class | Count | Percentage |
|---|---:|---:|
| Fraud | 5,760 | 1.21% |
| Legitimate | 469,954 | 98.79% |

### After SMOTE-NC

| Class | Count | Percentage |
|---|---:|---:|
| Fraud | 46,995 | 9.09% |
| Legitimate | 469,954 | 90.91% |

The test set is not resampled, which keeps evaluation realistic.

---

## Models Trained

Two models are trained and compared:

1. **Logistic Regression**
   - Used as a simple baseline model.
   - Requires feature scaling.

2. **Random Forest**
   - Used as the primary model.
   - Handles non-linear relationships well.
   - Provides feature importance for interpretation.

---

## Evaluation Metrics

Because fraud detection is an imbalanced classification problem, the project focuses on:

- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Confusion Matrix

PR-AUC and F1-score are especially important because they focus more directly on minority-class fraud detection performance.

---

## Model Results

### Random Forest

| Metric | Score |
|---|---:|
| ROC-AUC | 0.9952 |
| PR-AUC | 0.9137 |
| F1-score for Fraud | 0.8393 |
| Fraud Precision | 82% |
| Fraud Recall | 86% |

### Random Forest Confusion Matrix

| Result | Count |
|---|---:|
| True Positives — frauds caught | 1,243 |
| False Negatives — frauds missed | 197 |
| False Positives — false alerts | 279 |
| True Negatives — correct approvals | 117,210 |

The Random Forest model caught **86.3%** of fraudulent transactions while keeping the false alert rate at only **0.24%** of legitimate transactions.

---

### Logistic Regression

| Metric | Score |
|---|---:|
| ROC-AUC | 0.9970 |
| PR-AUC | 0.8768 |
| F1-score for Fraud | 0.7670 |

Although Logistic Regression achieved a slightly higher ROC-AUC, Random Forest achieved stronger PR-AUC and F1-score, making it more suitable for the fraud detection objective.

---

## Model Comparison

| Model | ROC-AUC | PR-AUC | F1 Fraud |
|---|---:|---:|---:|
| Random Forest | 0.9952 | 0.9137 | 0.8393 |
| Logistic Regression | 0.9970 | 0.8768 | 0.7670 |

### Final Model Recommendation

**Random Forest** is recommended as the final model because it delivers better fraud-focused performance, especially in terms of PR-AUC and F1-score. These metrics are more meaningful than accuracy for imbalanced fraud detection problems.

---

## Feature Importance Insights

The most important predictors include:

| Feature | Importance |
|---|---:|
| `merch_fraud_rate` | 26.4% |
| `merch_tx_count` | 19.2% |
| `cat_fraud_rate` | 15.3% |
| `log_amount` | 10.1% |
| `amount` | 9.5% |

### Business Interpretation

The strongest signal is merchant behavior. This suggests that who the customer transacts with is highly predictive of fraud risk. Transaction category and amount also provide strong signals.

---

## Business Recommendations

Based on the analysis and model results:

1. **Use Random Forest as the primary fraud detection model**
   - It provides strong fraud recall while maintaining manageable false alerts.

2. **Prioritize high-risk merchants and categories**
   - Merchant fraud rate and category fraud rate are among the strongest predictors.

3. **Monitor high-value transactions**
   - Fraudulent transactions have higher average amounts, making amount-based monitoring important.

4. **Use model probabilities for risk scoring**
   - Instead of only predicting fraud or legitimate, use probability thresholds to create risk levels:
     - Low risk
     - Medium risk
     - High risk

5. **Tune thresholds based on business cost**
   - If missing fraud is very expensive, lower the threshold to increase recall.
   - If manual review capacity is limited, raise the threshold to improve precision.

---

## How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/fraud-detection-case-study.git
cd fraud-detection-case-study
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

```bash
# Windows
venv\Scripts\activate
```

```bash
# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If you do not have a `requirements.txt` file yet, install the main packages manually:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn joblib jupyter
```

### 4. Open the Notebook

```bash
jupyter notebook fraud.ipynb
```

Then run the cells from top to bottom.

---

## Suggested `requirements`

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
joblib
jupyter
```

---

## Limitations

This project provides a strong analytical and modeling workflow, but several limitations should be considered:

- The model should be validated on newer unseen data before production use.
- Threshold tuning should be aligned with business cost and review capacity.
- Merchant and category fraud rates should be recalculated carefully over time to avoid leakage.
- Additional production monitoring would be required to detect model drift.
- Real-world deployment would need stronger governance, explainability, and alert management.

---

## Future Improvements

Potential improvements include:

- Hyperparameter tuning for Random Forest.
- Testing additional models such as XGBoost, LightGBM, or CatBoost.
- Adding threshold optimization based on business cost.
- Building a risk-score dashboard.
- Adding model explainability using SHAP.
- Creating an automated pipeline for retraining and monitoring.

---

## Final Conclusion

This project demonstrates a complete fraud detection workflow from raw data to business-ready model evaluation.  
The Random Forest model is the preferred solution because it achieves strong fraud detection performance, catches most fraudulent transactions, and keeps false alerts relatively low.

