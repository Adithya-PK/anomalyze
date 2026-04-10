# Anomalyze - Credit Card Fraud Detection using Anomaly Detection ML 

Anomalyze is an academic fraud detection project built with Machine Learning and Streamlit. It detects suspicious credit card transactions using a hybrid decision system:

- a Random Forest classifier for data-driven fraud probability
- a rule-based scoring engine for explainability
- a hybrid final score that combines both signals

The project does not use Kaggle data directly. Instead, it generates a realistic synthetic dataset inspired by common public fraud-detection workflows.

## Project Objective

The goal of this project is to build a clean, explainable, college-ready fraud detection dashboard that:

- uses realistic transaction fields only
- avoids fake hidden features such as `V1-V28`
- provides both single-transaction and batch analysis
- explains why a transaction is low, medium, or high risk
- combines model intelligence with business rules

## Core Features

The system uses these exact fields.

Numerical features:
- `amount`
- `time`
- `transaction_count_24h`
- `avg_amount_24h`

Categorical features:
- `merchant_type`
- `transaction_type`
- `device_type`
- `location_type`

Binary feature:
- `is_international`

Target:
- `fraud`

## Dataset Strategy

This project generates its own synthetic dataset in [train_model.py](C:/Users/apk05/OneDrive/Desktop/Anomalyze/train_model.py).

Dataset settings:
- total rows: `20,000`
- train split: `16,000`
- test split: `4,000`
- random state: `42`
- observed fraud rate: `4.32%`

The synthetic generator models realistic fraud patterns instead of random labels.

## Fraud Pattern Logic Used In Data Generation

Fraud probability is increased when one or more of these conditions occur:

- `amount > 10000`
- `is_international == 1`
- `transaction_type == "online"` and `location_type == "international"`
- transaction time is between late-night hours (`22:00` to `05:00`)
- `transaction_count_24h > 10`
- `amount` is much larger than `avg_amount_24h`

Additional smaller fraud adjustments are also applied for riskier merchant/location combinations.

## Model Training Pipeline

Training happens in [train_model.py](C:/Users/apk05/OneDrive/Desktop/Anomalyze/train_model.py).

Pipeline steps:
1. Generate a synthetic fraud dataset.
2. Split features and target.
3. Perform an 80/20 train-test split using stratification.
4. Standardize numerical features using `StandardScaler`.
5. One-hot encode categorical features using `OneHotEncoder(handle_unknown="ignore")`.
6. Keep binary feature `is_international` as numeric.
7. Concatenate all processed features.
8. Train a `RandomForestClassifier`.
9. Evaluate the model on the held-out test set.
10. Save artifacts for the Streamlit app.

Model configuration:

```python
RandomForestClassifier(
    n_estimators=150,
    max_depth=10,
    class_weight="balanced",
    random_state=42,
)
```

Saved files:
- [model.pkl](C:/Users/apk05/OneDrive/Desktop/Anomalyze/model.pkl)
- [scaler.pkl](C:/Users/apk05/OneDrive/Desktop/Anomalyze/scaler.pkl)
- [encoder.pkl](C:/Users/apk05/OneDrive/Desktop/Anomalyze/encoder.pkl)
- [feature_columns.pkl](C:/Users/apk05/OneDrive/Desktop/Anomalyze/feature_columns.pkl)

## Model Evaluation Metrics

These metrics come from the current deterministic model run on the test split.

Overall metrics:
- Accuracy: `0.9393`
- Precision (Fraud class): `0.3056`
- Recall (Fraud class): `0.3179`
- F1-score (Fraud class): `0.3116`
- ROC-AUC: `0.7507`

Confusion matrix:
- True Negative: `3702`
- False Positive: `125`
- False Negative: `118`
- True Positive: `55`

Class-wise performance:
- Normal precision: `0.9691`
- Normal recall: `0.9673`
- Normal F1-score: `0.9682`
- Fraud precision: `0.3056`
- Fraud recall: `0.3179`
- Fraud F1-score: `0.3116`

Interpretation:
- The model performs strongly on the majority normal class.
- Fraud detection is harder because fraud is intentionally rare and pattern-based.
- This is expected in an imbalanced fraud-detection setting.
- The rule engine and hybrid scoring are added to improve explainability and practical screening behavior.

## Most Important Model Features

Top learned feature importances from the trained Random Forest:

- `time`: `0.2059`
- `amount`: `0.2030`
- `avg_amount_24h`: `0.1227`
- `is_international`: `0.1107`
- `location_type_international`: `0.0982`
- `transaction_count_24h`: `0.0704`
- `location_type_same_city`: `0.0551`
- `location_type_different_city`: `0.0200`
- `merchant_type_grocery`: `0.0171`
- `merchant_type_electronics`: `0.0158`

This indicates that transaction timing, amount behavior, recent average amount, and international context are the strongest predictive signals in this project.

## What Each Column Means

`amount`
- Current transaction amount.
- Higher values increase fraud risk, especially when far above the user's average.

`time`
- Hour-style transaction time from `0` to `23.99`.
- Late-night transactions are treated as riskier.

`transaction_count_24h`
- Number of recent transactions in the last 24 hours.
- High counts can indicate suspicious rapid activity.

`avg_amount_24h`
- User's average transaction amount in the last 24 hours.
- Used to detect behavioral anomalies when the current amount is much higher than usual.

`merchant_type`
- Category of merchant.
- `electronics` and `travel` are treated as relatively riskier than everyday categories.

`transaction_type`
- Whether the transaction is `online` or `POS`.
- Online transactions are generally more suspicious than point-of-sale transactions.

`device_type`
- Transaction device such as `mobile` or `desktop`.
- Used as a supporting contextual signal.

`location_type`
- Whether the transaction happens in `same_city`, `different_city`, or `international` context.
- International activity is a strong fraud indicator.

`is_international`
- Binary indicator (`0` or `1`).
- Gives the model and rule engine a direct risk signal for international transactions.

## Rule-Based Scoring System

The rule engine gives transparent fraud points.

Rules:
- `amount > 10000` -> `+20`
- `is_international == 1` -> `+25`
- `online + international` -> `+15`
- late-night transaction -> `+10`
- high transaction frequency -> `+10`
- amount much larger than recent average -> `+20`

Rule score range:
- `0` to `100`

## Hybrid Scoring Logic

The app does not rely on the ML model alone.

It calculates:

```text
base_score = (ml_probability * 0.55) + (rule_score / 100 * 0.45)
final_score = base_score + escalation_bonus
```

The severe-risk escalation bonus is applied when strong fraud signals occur together, such as:
- high amount + international
- online + international location
- very high frequency + amount far above recent average

This helps obviously suspicious transactions reach `High Risk` more often.

Risk bands:
- `0-30` -> Low
- `30-70` -> Medium
- `70-100` -> High

Prediction rule:
- `> 50` -> Fraud
- `<= 50` -> Normal

## Short Risk Explanation In Batch Mode

The batch output includes a short reason column with compact explanations such as:
- `Low risk normal pattern`
- `Rules high model moderate`
- `Multiple severe fraud signals`
- `Strong international fraud pattern`

This is meant to provide a quick one-line interpretation without showing the full long explanation text in the table.

## Streamlit Dashboard Features

The dashboard is implemented in [app.py](C:/Users/apk05/OneDrive/Desktop/Anomalyze/app.py).

Single transaction mode:
- manual form input
- anomaly score
- risk level
- prediction
- confidence
- rule/ML decision summary
- explanation panel
- progress bar
- feature importance chart
- risk distribution chart

Batch mode:
- CSV upload
- full row-wise fraud scoring
- risk-colored results table
- short risk explanation column
- summary cards
- risk distribution charts
- feature importance section
- downloadable results CSV

## Files In The Project

Main files:
- [train_model.py](C:/Users/apk05/OneDrive/Desktop/Anomalyze/train_model.py)
- [app.py](C:/Users/apk05/OneDrive/Desktop/Anomalyze/app.py)
- [requirements.txt](C:/Users/apk05/OneDrive/Desktop/Anomalyze/requirements.txt)
- [README.md](C:/Users/apk05/OneDrive/Desktop/Anomalyze/README.md)

Artifacts:
- [model.pkl](C:/Users/apk05/OneDrive/Desktop/Anomalyze/model.pkl)
- [scaler.pkl](C:/Users/apk05/OneDrive/Desktop/Anomalyze/scaler.pkl)
- [encoder.pkl](C:/Users/apk05/OneDrive/Desktop/Anomalyze/encoder.pkl)
- [feature_columns.pkl](C:/Users/apk05/OneDrive/Desktop/Anomalyze/feature_columns.pkl)

Sample files:
- [sample_single_transaction.csv](C:/Users/apk05/OneDrive/Desktop/Anomalyze/sample_single_transaction.csv)
- [sample.csv](C:/Users/apk05/OneDrive/Desktop/Anomalyze/sample.csv)

## How To Run The Project

```bash
cd C:\Users\apk05\OneDrive\Desktop\Anomalyze
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## Batch CSV Format

Your CSV must use exactly these columns:

```csv
amount,time,transaction_count_24h,avg_amount_24h,merchant_type,transaction_type,device_type,location_type,is_international
```

Example:

```csv
amount,time,transaction_count_24h,avg_amount_24h,merchant_type,transaction_type,device_type,location_type,is_international
320.00,13.25,2,280.00,grocery,POS,mobile,same_city,0
22000.00,23.40,14,3200.00,electronics,online,desktop,international,1
```

## Notes

- This project does not download the Kaggle credit card fraud dataset.
- Kaggle projects are referenced only as inspiration for academic structure.
- The model is trained entirely on synthetic but realistic fraud patterns.
- The dashboard predictions use only the visible user inputs and saved preprocessing artifacts.
