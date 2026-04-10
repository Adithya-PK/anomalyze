"""
Train a hybrid-ready fraud detection model for the Anomalyze dashboard.

This project uses a synthetic dataset inspired by the structure of public
credit card fraud projects such as the Kaggle credit card fraud dataset, but
it does not download or reuse Kaggle data.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
NUM_ROWS = 20_000

NUMERICAL_FEATURES = [
    "amount",
    "time",
    "transaction_count_24h",
    "avg_amount_24h",
]
CATEGORICAL_FEATURES = [
    "merchant_type",
    "transaction_type",
    "device_type",
    "location_type",
]
BINARY_FEATURES = ["is_international"]
TARGET_COLUMN = "fraud"

MERCHANT_TYPES = ["food", "electronics", "travel", "grocery"]
TRANSACTION_TYPES = ["online", "POS"]
DEVICE_TYPES = ["mobile", "desktop"]
LOCATION_TYPES = ["same_city", "different_city", "international"]


def generate_synthetic_dataset(n_rows: int = NUM_ROWS, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Create a realistic fraud dataset with a 3-5% fraud rate."""
    rng = np.random.default_rng(seed)

    merchant_type = rng.choice(
        MERCHANT_TYPES,
        size=n_rows,
        p=[0.28, 0.19, 0.14, 0.39],
    )
    transaction_type = rng.choice(
        TRANSACTION_TYPES,
        size=n_rows,
        p=[0.42, 0.58],
    )
    device_type = rng.choice(
        DEVICE_TYPES,
        size=n_rows,
        p=[0.63, 0.37],
    )
    location_type = rng.choice(
        LOCATION_TYPES,
        size=n_rows,
        p=[0.77, 0.18, 0.05],
    )
    is_international = (location_type == "international").astype(int)

    hour_of_day = np.clip(rng.normal(loc=13, scale=6, size=n_rows), 0, 23.99)
    minute_offset = rng.uniform(0, 0.99, size=n_rows)
    time = np.round(hour_of_day + minute_offset, 2)

    merchant_multiplier = pd.Series(merchant_type).map(
        {
            "food": 0.7,
            "electronics": 1.6,
            "travel": 2.5,
            "grocery": 0.9,
        }
    ).to_numpy()
    location_multiplier = pd.Series(location_type).map(
        {
            "same_city": 1.0,
            "different_city": 1.25,
            "international": 1.8,
        }
    ).to_numpy()

    transaction_count_24h = np.maximum(
        1,
        rng.poisson(
            lam=2.5
            + (transaction_type == "online") * 1.1
            + (location_type == "different_city") * 0.8
            + (location_type == "international") * 1.3,
            size=n_rows,
        ),
    )

    avg_amount_24h = np.clip(
        rng.lognormal(mean=4.1, sigma=0.7, size=n_rows) * merchant_multiplier,
        50,
        15_000,
    )

    amount = np.clip(
        avg_amount_24h
        * rng.lognormal(mean=0.05, sigma=0.75, size=n_rows)
        * location_multiplier
        * np.where(transaction_type == "online", 1.08, 0.96),
        10,
        60_000,
    )

    amount_ratio = amount / np.maximum(avg_amount_24h, 1)
    is_night = ((time >= 22) | (time <= 5)).astype(int)

    fraud_probability = (
        0.014
        + (amount > 10_000) * 0.09
        + (is_international == 1) * 0.16
        + ((transaction_type == "online") & (location_type == "international")) * 0.24
        + (is_night == 1) * 0.05
        + (transaction_count_24h > 10) * 0.08
        + (amount_ratio > 4.0) * 0.11
        + ((transaction_type == "online") & (location_type == "different_city")) * 0.025
        + (merchant_type == "electronics") * 0.014
        + (merchant_type == "travel") * 0.02
        + ((transaction_type == "POS") & (location_type == "same_city") & (amount < avg_amount_24h * 1.2)) * -0.008
        + ((merchant_type == "grocery") & (amount < 500) & (transaction_count_24h <= 3)) * -0.006
    )
    fraud_probability = np.clip(fraud_probability, 0.001, 0.95)
    fraud = rng.binomial(1, fraud_probability)

    dataset = pd.DataFrame(
        {
            "amount": np.round(amount, 2),
            "time": np.round(time, 2),
            "transaction_count_24h": transaction_count_24h.astype(int),
            "avg_amount_24h": np.round(avg_amount_24h, 2),
            "merchant_type": merchant_type,
            "transaction_type": transaction_type,
            "device_type": device_type,
            "location_type": location_type,
            "is_international": is_international.astype(int),
            "fraud": fraud.astype(int),
        }
    )
    return dataset


def preprocess_training_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
):
    """Fit preprocessors on the train split and transform both splits."""
    scaler = StandardScaler()
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    x_train_num = scaler.fit_transform(train_df[NUMERICAL_FEATURES])
    x_test_num = scaler.transform(test_df[NUMERICAL_FEATURES])

    x_train_cat = encoder.fit_transform(train_df[CATEGORICAL_FEATURES])
    x_test_cat = encoder.transform(test_df[CATEGORICAL_FEATURES])

    x_train_bin = train_df[BINARY_FEATURES].to_numpy(dtype=float)
    x_test_bin = test_df[BINARY_FEATURES].to_numpy(dtype=float)

    x_train = np.hstack([x_train_num, x_train_bin, x_train_cat])
    x_test = np.hstack([x_test_num, x_test_bin, x_test_cat])

    feature_columns = (
        NUMERICAL_FEATURES
        + BINARY_FEATURES
        + encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    )
    return x_train, x_test, scaler, encoder, feature_columns


def train_model(output_dir: str = ".") -> None:
    """Train the Random Forest model and export all artifacts."""
    output_path = Path(output_dir)
    dataset = generate_synthetic_dataset()

    fraud_rate = dataset[TARGET_COLUMN].mean() * 100
    print(f"Generated dataset with shape: {dataset.shape}")
    print(f"Fraud rate: {fraud_rate:.2f}%")

    features = dataset[NUMERICAL_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES]
    target = dataset[TARGET_COLUMN]

    x_train_df, x_test_df, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    x_train, x_test, scaler, encoder, feature_columns = preprocess_training_data(
        x_train_df,
        x_test_df,
    )

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, probabilities)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

    output_path.mkdir(parents=True, exist_ok=True)

    with (output_path / "model.pkl").open("wb") as file:
        pickle.dump(model, file)
    with (output_path / "scaler.pkl").open("wb") as file:
        pickle.dump(scaler, file)
    with (output_path / "encoder.pkl").open("wb") as file:
        pickle.dump(encoder, file)
    with (output_path / "feature_columns.pkl").open("wb") as file:
        pickle.dump(feature_columns, file)

    print("Saved: model.pkl, scaler.pkl, encoder.pkl, feature_columns.pkl")


if __name__ == "__main__":
    train_model()