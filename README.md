# Fraud Detection Case Study

A professional machine learning case study for detecting fraudulent financial transactions using Python and scikit-learn.

This project follows an end-to-end workflow: data loading, data cleaning, exploratory data analysis, feature engineering, leakage-safe train/test preparation, class imbalance handling with SMOTE-NC, model training, model evaluation, feature importance analysis, and business recommendations.


---

## Project Objective

The objective of this project is to build a machine learning model that can classify financial transactions as either:

| Class | Meaning |
|---|---|
| `0` | Legitimate transaction |
| `1` | Fraudulent transaction |

Fraud detection is an imbalanced classification problem. In this dataset, fraud transactions are rare, so accuracy alone is not enough. The project focuses on fraud-relevant metrics such as precision, recall, F1-score, ROC-AUC, and PR-AUC.

---

## Repository Structure

```text
fraud-detection-case-study/
│
├── fraud.ipynb      # Main Jupyter Notebook
├── fraud.csv        # Transaction dataset required by the notebook
└── README.md        # Project documentation
```

> Important: The notebook uses:
>
> ```python
> DATA_PATH = "fraud.csv"
> ```
>
> Therefore, the dataset file must be named `fraud.csv` and placed in the same folder as `fraud.ipynb`.

---

## Dataset Overview

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
| `fraud` | Target variable |

### Target Distribution

| Class | Count | Percentage |
|---|---:|---:|
| Legitimate transactions | 587,443 | 98.79% |
| Fraudulent transactions | 7,200 | 1.21% |

The dataset is highly imbalanced. A naive model that predicts every transaction as legitimate would achieve **98.79% accuracy**, but it would miss every fraud case. For this reason, the project uses fraud-focused evaluation metrics.

---

## Workflow Summary

### 1. Data Loading and Cleaning

The notebook loads the dataset using pandas and performs initial cleaning steps, including:

- Loading `fraud.csv`.
- Removing quote artifacts from categorical columns.
- Checking data types.
- Checking missing values.
- Reviewing dataset shape and memory usage.

The following categorical columns are cleaned by removing surrounding quote characters:

```python
customer, age, gender, zipcodeOri, merchant, zipMerchant, category
```

---

### 2. Exploratory Data Analysis

The analysis explores fraud behavior across several areas:

- Class imbalance.
- Transaction amount distribution.
- Fraud rate by transaction category.
- Fraud patterns by hour and day.
- Fraud rate by gender and age.
- Merchant-level fraud behavior.
- Customer-level transaction behavior.

Key findings from the notebook:

- Fraudulent transactions are rare but have much higher average transaction amounts.
- Transaction amount is a strong fraud signal.
- Merchant behavior is highly predictive of fraud risk.
- Category-level fraud rates provide useful risk information.
- The dataset covers steps from `0` to `179`, approximately **7.5 days**.

---

### 3. Preprocessing

The notebook prepares the dataset by:

- Dropping zero-variance columns:
  - `zipcodeOri`
  - `zipMerchant`
- Encoding categorical variables:
  - `age`
  - `gender`
  - `category`
- Creating encoded columns:
  - `age_enc`
  - `gender_enc`
  - `category_enc`

The zero-variance columns are removed because both contain only one unique value and do not add predictive value.

---

### 4. Feature Engineering

The final modeling dataset contains **15 features**.

### Feature Groups

| Feature Group | Features |
|---|---|
| Raw numeric | `step` |
| Encoded categorical | `age_enc`, `gender_enc`, `category_enc` |
| Temporal | `hour_of_day`, `day_of_week`, `is_night` |
| Amount-based | `amount`, `log_amount` |
| Customer behavior | `cust_tx_count`, `cust_avg_amount`, `cust_max_amount` |
| Merchant and category risk | `merch_fraud_rate`, `merch_tx_count`, `cat_fraud_rate` |

### Leakage-Safe Aggregation

The notebook creates historical aggregate features such as:

- `merch_fraud_rate`
- `cat_fraud_rate`
- `cust_tx_count`
- `cust_avg_amount`
- `cust_max_amount`
- `merch_tx_count`

To reduce target leakage, aggregate mappings are calculated using the training set and then applied to both training and test sets.

---

## Train/Test Split

The notebook uses a stratified train/test split.

| Dataset | Rows | Fraud Rate |
|---|---:|---:|
| Training set | 475,714 | 1.21% |
| Test set | 118,929 | 1.21% |

Fraud counts:

| Dataset | Fraud Count |
|---|---:|
| Training set | 5,760 |
| Test set | 1,440 |

---

## Class Imbalance Handling

Because the fraud class represents only **1.21%** of the dataset, the notebook applies **SMOTE-NC** to the training data only.

SMOTE-NC is used because the feature set contains both numeric and categorical/discrete features.

### Before SMOTE-NC

| Class | Count | Percentage |
|---|---:|---:|
| Fraud | 5,760 | 1.21% |
| Legitimate | 469,954 | 98.79% |
| Total | 475,714 | 100% |

### After SMOTE-NC

| Class | Count | Percentage |
|---|---:|---:|
| Fraud | 46,995 | 9.09% |
| Legitimate | 469,954 | 90.91% |
| Total | 516,949 | 100% |

The test set is not resampled, which keeps the evaluation realistic.

---

## Models Trained

The notebook trains and compares two models.

### 1. Random Forest

The primary model is a Random Forest trained on the SMOTE-NC resampled training set.

Main configuration:

```python
RandomForestClassifier(
    n_estimators=100,
    random_state=SEED,
    n_jobs=-1
)
```

### 2. Logistic Regression

Logistic Regression is used as a baseline model.

The notebook scales the features before training Logistic Regression using `StandardScaler`.

---

## Evaluation Metrics

The project uses the following metrics:

| Metric | Why It Matters |
|---|---|
| Precision | Measures how many predicted fraud cases were actually fraud |
| Recall | Measures how many real fraud cases were caught |
| F1-score | Balances precision and recall |
| ROC-AUC | Measures overall ranking ability |
| PR-AUC | Better suited for imbalanced fraud detection |
| Confusion Matrix | Shows fraud caught, fraud missed, and false alerts |

---

## Model Results

### Random Forest Results

| Metric | Score |
|---|---:|
| ROC-AUC | 0.9952 |
| PR-AUC | 0.9137 |
| F1-score for fraud | 0.8393 |
| Fraud precision | 82% |
| Fraud recall | 86% |

### Random Forest Confusion Matrix

| Result | Count |
|---|---:|
| True Positives — frauds caught | 1,243 |
| False Negatives — frauds missed | 197 |
| False Positives — false alerts | 279 |
| True Negatives — correct approvals | 117,210 |

The Random Forest model caught **86.32%** of fraudulent transactions and produced a false alert rate of **0.24%** on legitimate transactions.

---

### Logistic Regression Results

| Metric | Score |
|---|---:|
| ROC-AUC | 0.9970 |
| PR-AUC | 0.8768 |
| F1-score for fraud | 0.7670 |

Logistic Regression achieved a slightly higher ROC-AUC, but it had lower PR-AUC and lower fraud F1-score than Random Forest.

---

## Model Comparison

| Model | ROC-AUC | PR-AUC | F1 Fraud |
|---|---:|---:|---:|
| Random Forest | 0.9952 | 0.9137 | 0.8393 |
| Logistic Regression | 0.9970 | 0.8768 | 0.7670 |

## Final Model Recommendation

The recommended model is **Random Forest**.

Although Logistic Regression achieved a slightly higher ROC-AUC, Random Forest performed better on the metrics that matter more for fraud detection:

- Higher PR-AUC.
- Higher fraud F1-score.
- Strong fraud recall.
- Manageable false alert rate.

For an imbalanced fraud detection problem, PR-AUC and F1-score are more useful than accuracy alone.

---

## Feature Importance Analysis

The notebook also trains a separate Random Forest model for feature importance analysis using:

```python
RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)
```

This section is used for model interpretation and business insights.

### Top Feature Importances

| Feature | Importance |
|---|---:|
| `merch_fraud_rate` | 26.4% |
| `merch_tx_count` | 19.2% |
| `cat_fraud_rate` | 15.3% |
| `log_amount` | 10.1% |
| `amount` | 9.5% |

### Interpretation

The strongest fraud signal is merchant behavior. This means the merchant involved in the transaction is highly predictive of fraud risk. Category risk and transaction amount are also important predictors.

---

## Business Recommendations

Based on the model results and analysis:

1. **Use Random Forest as the primary fraud detection model.**  
   It provides the best balance between fraud detection and manageable false alerts.

2. **Prioritize high-risk merchants.**  
   Merchant-level fraud rate is the strongest predictor in the notebook.

3. **Monitor high-risk transaction categories.**  
   Some categories have much higher fraud rates than others.

4. **Use transaction amount as a risk signal.**  
   Fraudulent transactions have a much higher average amount than legitimate transactions.

5. **Use probability scores instead of only hard predictions.**  
   Model probabilities can support risk levels such as low, medium, and high risk.

6. **Tune the decision threshold based on business needs.**  
   Lower thresholds increase fraud recall, while higher thresholds reduce false alerts.

---

## How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/fraud-detection-case-study.git
cd fraud-detection-case-study
```

Replace `your-username` with your GitHub username.

### 2. Install Required Packages

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn joblib jupyter
```

### 3. Open the Notebook

```bash
jupyter notebook fraud.ipynb
```

### 4. Run the Notebook

Run all cells from top to bottom.

Make sure that `fraud.csv` is in the same folder as `fraud.ipynb`.

---

## Requirements

The notebook uses the following Python libraries:

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

This project is suitable as a case study and analytical prototype. Before production use, the following points should be considered:

- Validate the model on newer unseen data.
- Tune the classification threshold using business cost assumptions.
- Monitor model performance over time.
- Recalculate fraud-rate features carefully to avoid leakage.
- Add stronger explainability for operational fraud review.
- Build a retraining and monitoring pipeline.

---

## Future Improvements

Potential future enhancements include:

- Hyperparameter tuning for Random Forest.
- Testing advanced gradient boosting models such as XGBoost, LightGBM, or CatBoost.
- Adding SHAP-based explainability.
- Building a fraud risk dashboard.
- Creating an automated model training pipeline.
- Adding threshold optimization based on fraud loss and review capacity.

---

## Final Conclusion

This project demonstrates a complete fraud detection machine learning workflow. The notebook shows how to clean transaction data, analyze fraud patterns, engineer predictive features, handle class imbalance, train models, evaluate results, and translate model outputs into business recommendations.

The final recommended model is **Random Forest** because it achieved the strongest fraud-focused performance, with a fraud F1-score of **0.8393**, PR-AUC of **0.9137**, and fraud recall of **86.32%**.
