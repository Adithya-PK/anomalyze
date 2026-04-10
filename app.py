"""
Anomalyze - Credit Card Fraud Detection Dashboard

Hybrid scoring:
1. Random Forest probability
2. Rule-based explainability score
"""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
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
REQUIRED_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES

DARK_BG = "#0a0e1a"
CARD_BG = "#10192b"
SIDEBAR_BG = "#0b1221"
BORDER = "#1f2d45"
TEXT = "#c8d6f0"
MUTED = "#5e7297"
LOW = "#45e17a"
MEDIUM = "#ffd12a"
HIGH = "#ff4d5e"
ACCENT = "#38a3ff"


st.set_page_config(
    page_title="Anomalyze",
    page_icon="AI",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Syne:wght@700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'JetBrains Mono', monospace;
        background: #0a0e1a;
        color: #c8d6f0;
    }

    .stApp {
        background:
            radial-gradient(circle at top right, rgba(65, 84, 255, 0.10), rgba(10, 14, 26, 0) 28%),
            radial-gradient(circle at top left, rgba(46, 196, 182, 0.07), rgba(10, 14, 26, 0) 24%),
            #0a0e1a;
    }

    .block-container {
        padding-top: 4.75rem;
        padding-bottom: 2.2rem;
    }

    section[data-testid="stSidebar"] {
        background: #0b1221;
        border-right: 1px solid #1f2d45;
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.3rem;
    }

    h1, h2, h3, h4, h5, h6, label, p, span, div {
        color: #c8d6f0;
    }

    .main-title {
        font-family: 'Syne', sans-serif;
        font-size: clamp(2.2rem, 4.2vw, 3.2rem);
        font-weight: 800;
        line-height: 1.05;
        letter-spacing: -1px;
        margin: 0.35rem 0 0 0;
        max-width: 100%;
        word-break: break-word;
        overflow-wrap: anywhere;
        background: linear-gradient(90deg, #48b5ff 0%, #6ea5ff 35%, #b36bff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .sidebar-logo {
        font-family: 'Syne', sans-serif;
        font-size: 2.6rem;
        font-weight: 800;
        line-height: 1;
        margin: 0 0 0.35rem 0;
        letter-spacing: -1px;
        background: linear-gradient(90deg, #48b5ff 0%, #6ea5ff 40%, #b36bff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .subtitle {
        margin-top: 0.35rem;
        margin-bottom: 1.7rem;
        color: #4d6187;
        font-size: 0.8rem;
        letter-spacing: 4px;
        text-transform: uppercase;
    }

    .section-line {
        border: none;
        border-top: 1px solid #1f2d45;
        margin: 1.1rem 0 2rem 0;
    }

    .metric-card {
        background: #10192b;
        border: 1px solid #19304f;
        border-radius: 14px;
        padding: 1.25rem 1rem;
        text-align: center;
        min-height: 128px;
        box-shadow: inset 0 0 0 1px rgba(56, 163, 255, 0.02);
    }

    .metric-label {
        color: #41557a;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 3px;
        margin-bottom: 1rem;
    }

    .metric-value {
        font-family: 'Syne', sans-serif;
        font-size: 2.3rem;
        font-weight: 800;
        line-height: 1;
    }

    .panel {
        background: #0d1525;
        border: 1px solid #1b2942;
        border-radius: 14px;
        padding: 1rem 1.1rem;
    }

    .alert-box {
        border-radius: 12px;
        padding: 1rem 1.1rem;
        border: 1px solid transparent;
        margin: 0.8rem 0 1rem 0;
        font-weight: 600;
    }

    .alert-low {
        background: rgba(69, 225, 122, 0.22);
        border-color: rgba(69, 225, 122, 0.18);
        color: #6ff09a;
    }

    .alert-medium {
        background: rgba(255, 209, 42, 0.18);
        border-color: rgba(255, 209, 42, 0.22);
        color: #ffd95b;
    }

    .alert-high {
        background: rgba(255, 77, 94, 0.18);
        border-color: rgba(255, 77, 94, 0.22);
        color: #ff7583;
    }

    .confidence-chip {
        display: inline-block;
        padding: 0.18rem 0.5rem;
        border-radius: 6px;
        background: rgba(69, 225, 122, 0.14);
        color: #73f39d;
        font-weight: 700;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-baseweb="base-input"] > div,
    .stNumberInput > div > div,
    .stTextInput > div > div {
        background: #232634 !important;
        border: 1px solid #30384c !important;
        color: #e6eefc !important;
        border-radius: 10px !important;
    }

    .stSlider [data-baseweb="slider"] {
        padding-top: 0.5rem;
    }

    .stButton button,
    .stDownloadButton button {
        background: #171e2b;
        color: #d9e5ff;
        border: 1px solid #343d52;
        border-radius: 10px;
        min-height: 44px;
        font-weight: 700;
    }

    .stButton button:hover,
    .stDownloadButton button:hover {
        border-color: #5485ff;
        color: white;
    }

    .stRadio label {
        color: #c8d6f0 !important;
    }

    .stProgress > div > div > div > div {
        background: #2693ff;
    }

    .stProgress > div > div > div {
        background: #2a3040;
    }

    .stAlert {
        background: #132846;
        border: 1px solid #22456d;
        color: #84b9ff;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #1f2d45;
        border-radius: 14px;
        overflow: hidden;
    }

    @media (max-width: 1200px) {
        .main-title {
            font-size: 2.4rem;
        }
    }

    @media (max-width: 768px) {
        .block-container {
            padding-top: 5.4rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts():
    with (BASE_DIR / "model.pkl").open("rb") as file:
        model = pickle.load(file)
    with (BASE_DIR / "scaler.pkl").open("rb") as file:
        scaler = pickle.load(file)
    with (BASE_DIR / "encoder.pkl").open("rb") as file:
        encoder = pickle.load(file)
    with (BASE_DIR / "feature_columns.pkl").open("rb") as file:
        feature_columns = pickle.load(file)
    return model, scaler, encoder, feature_columns


def prepare_features(frame: pd.DataFrame, scaler, encoder) -> np.ndarray:
    numeric = scaler.transform(frame[NUMERICAL_FEATURES])
    binary = frame[BINARY_FEATURES].to_numpy(dtype=float)
    categorical = encoder.transform(frame[CATEGORICAL_FEATURES])
    return np.hstack([numeric, binary, categorical])


def rule_based_score(input_data: pd.Series | dict) -> tuple[int, list[str]]:
    row = pd.Series(input_data)
    score = 0
    reasons: list[str] = []

    if row["amount"] > 10_000:
        score += 20
        reasons.append("High transaction amount increased risk.")
    if int(row["is_international"]) == 1:
        score += 25
        reasons.append("International transaction detected.")
    if row["transaction_type"] == "online" and row["location_type"] == "international":
        score += 15
        reasons.append("Online transaction from an international location is very high risk.")
    if row["time"] >= 22 or row["time"] <= 5:
        score += 10
        reasons.append("Late-night transaction timing increased suspicion.")
    if row["transaction_count_24h"] > 10:
        score += 10
        reasons.append("High transaction frequency in the last 24 hours is suspicious.")
    if row["amount"] > max(row["avg_amount_24h"] * 3, row["avg_amount_24h"] + 5000):
        score += 20
        reasons.append("Current amount is unusually high compared with the recent average.")

    if not reasons:
        reasons.append("No major rule-based red flags were triggered.")

    return min(score, 100), reasons


def final_decision(
    ml_probability: float,
    rule_score: int,
    row: pd.Series | dict | None = None,
) -> dict[str, float | int | str]:
    base_score = ((ml_probability * 0.55) + ((rule_score / 100) * 0.45)) * 100

    escalation_bonus = 0
    if row is not None:
        record = pd.Series(row)
        severe_combo = (
            (record["amount"] > 10_000 and int(record["is_international"]) == 1)
            or (record["transaction_type"] == "online" and record["location_type"] == "international")
            or (
                record["transaction_count_24h"] > 10
                and record["amount"] > max(record["avg_amount_24h"] * 3, record["avg_amount_24h"] + 5000)
            )
        )
        if severe_combo and rule_score >= 70:
            escalation_bonus += 10

    if rule_score >= 90 and ml_probability >= 0.12:
        escalation_bonus += 10
    if rule_score >= 80 and ml_probability >= 0.20:
        escalation_bonus += 5

    final_score = round(min(base_score + escalation_bonus, 100), 2)

    if final_score < 30:
        risk_level = "Low"
    elif final_score < 70:
        risk_level = "Medium"
    else:
        risk_level = "High"

    prediction = "Fraud" if final_score > 50 else "Normal"
    confidence = round(max(final_score, 100 - final_score), 2)

    return {
        "final_score": final_score,
        "risk_level": risk_level,
        "prediction": prediction,
        "confidence": confidence,
        "escalation_bonus": escalation_bonus,
    }


def explain_prediction(
    row: pd.Series,
    ml_probability: float,
    rule_reasons: list[str],
    escalation_bonus: int = 0,
) -> list[str]:
    explanations = list(rule_reasons)
    if ml_probability >= 0.75:
        explanations.append("The Random Forest model found a strong fraud pattern match.")
    elif ml_probability >= 0.45:
        explanations.append("The model detected a moderate fraud pattern that needs review.")
    else:
        explanations.append("The model probability stayed closer to normal transaction behavior.")

    if row["location_type"] == "different_city":
        explanations.append("A different-city transaction added moderate location risk.")
    if row["merchant_type"] in {"electronics", "travel"}:
        explanations.append(f"{row['merchant_type'].title()} purchases are treated as higher-risk merchant activity.")
    if row["device_type"] == "desktop" and row["transaction_type"] == "online":
        explanations.append("Desktop-based online activity is less common than mobile in this dataset profile.")
    if escalation_bonus > 0:
        explanations.append("The hybrid scorer added a severe-risk escalation bonus because multiple strong fraud signals appeared together.")

    return explanations


def short_risk_reason(
    row: pd.Series,
    risk_level: str,
    rule_score: int,
    ml_probability: float,
    escalation_bonus: int = 0,
) -> str:
    if risk_level == "High":
        if escalation_bonus > 0:
            return "Multiple severe fraud signals"
        if int(row["is_international"]) == 1:
            return "Strong international fraud pattern"
        return "High anomaly score detected"

    if risk_level == "Medium":
        if rule_score >= 70:
            return "Rules high model moderate"
        if row["transaction_type"] == "online":
            return "Online activity needs review"
        return "Moderate anomaly score detected"

    if rule_score == 0 and ml_probability < 0.30:
        return "Low risk normal pattern"
    if row["location_type"] == "same_city":
        return "Same city regular activity"
    return "Minor signals only detected"


def score_transactions(frame: pd.DataFrame, model, scaler, encoder) -> pd.DataFrame:
    prepared = prepare_features(frame, scaler, encoder)
    ml_probabilities = model.predict_proba(prepared)[:, 1]

    records = []
    for idx, (_, row) in enumerate(frame.iterrows()):
        rule_score, rule_reasons = rule_based_score(row)
        decision = final_decision(float(ml_probabilities[idx]), rule_score, row)
        explanations = explain_prediction(
            row,
            float(ml_probabilities[idx]),
            rule_reasons,
            int(decision["escalation_bonus"]),
        )
        short_reason = short_risk_reason(
            row,
            str(decision["risk_level"]),
            rule_score,
            float(ml_probabilities[idx]),
            int(decision["escalation_bonus"]),
        )
        records.append(
            {
                **row.to_dict(),
                "ml_probability": round(float(ml_probabilities[idx]) * 100, 2),
                "rule_score": rule_score,
                "anomaly_score": decision["final_score"],
                "risk_level": decision["risk_level"],
                "risk_reason": short_reason,
                "prediction": decision["prediction"],
                "confidence": decision["confidence"],
                "escalation_bonus": int(decision["escalation_bonus"]),
                "explanations": " | ".join(explanations),
            }
        )

    return pd.DataFrame(records)


def metric_value_color(risk_level: str) -> str:
    return {"Low": LOW, "Medium": MEDIUM, "High": HIGH}.get(risk_level, TEXT)


def style_predictions(frame: pd.DataFrame):
    def highlight_risk(row):
        color_map = {
            "Low": "background-color: rgba(69, 225, 122, 0.16); color: #70f29b;",
            "Medium": "background-color: rgba(255, 209, 42, 0.16); color: #ffd95b;",
            "High": "background-color: rgba(255, 77, 94, 0.16); color: #ff7c8a;",
        }
        style = color_map.get(row["risk_level"], "")
        return [style if col in {"risk_level", "prediction", "anomaly_score"} else "" for col in row.index]

    return frame.style.apply(highlight_risk, axis=1).format(
        {
            "amount": "{:.2f}",
            "time": "{:.2f}",
            "avg_amount_24h": "{:.2f}",
            "ml_probability": "{:.2f}",
            "anomaly_score": "{:.2f}",
            "confidence": "{:.2f}",
        }
    )


def render_feature_importance(model, feature_columns: list[str]) -> None:
    importance = (
        pd.DataFrame({"feature": feature_columns, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .head(10)
        .sort_values("importance")
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(CARD_BG)
    ax.barh(importance["feature"], importance["importance"], color=ACCENT)
    ax.set_title("Feature Importance", color=TEXT, fontsize=12, pad=10)
    ax.set_xlabel("Importance", color=TEXT)
    ax.tick_params(colors=TEXT)
    for spine in ax.spines.values():
        spine.set_color(BORDER)
    ax.grid(axis="x", color=BORDER, alpha=0.55)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_risk_distribution(results: pd.DataFrame) -> None:
    distribution = (
        results["risk_level"]
        .value_counts()
        .reindex(["Low", "Medium", "High"], fill_value=0)
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.patch.set_facecolor(DARK_BG)

    ax1.set_facecolor(CARD_BG)
    bars = ax1.bar(distribution.index, distribution.values, color=[LOW, MEDIUM, HIGH], width=0.56)
    ax1.set_title("Transactions by Risk Level", color=TEXT, fontsize=12)
    ax1.set_ylabel("Count", color=TEXT)
    ax1.tick_params(colors=TEXT)
    for spine in ax1.spines.values():
        spine.set_color(BORDER)
    ax1.grid(axis="y", color=BORDER, alpha=0.55)
    for bar in bars:
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(distribution.max() * 0.02, 0.5),
            f"{int(bar.get_height())}",
            ha="center",
            color=TEXT,
            fontsize=10,
            fontweight="bold",
        )

    ax2.set_facecolor(CARD_BG)
    non_zero = distribution[distribution > 0]
    if len(non_zero):
        ax2.pie(
            non_zero.values,
            labels=non_zero.index.tolist(),
            colors=[{"Low": LOW, "Medium": MEDIUM, "High": HIGH}[label] for label in non_zero.index],
            autopct="%1.1f%%",
            textprops={"color": TEXT, "fontsize": 10},
            wedgeprops={"edgecolor": DARK_BG, "linewidth": 2},
        )
        ax2.set_title("Risk Distribution Share", color=TEXT, fontsize=12)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_alert(risk_level: str, score: float) -> None:
    text = {
        "Low": f"Low Risk - Transaction appears normal (Score: {score:.2f}%)",
        "Medium": f"Medium Risk - Unusual behavior detected. Review recommended (Score: {score:.2f}%)",
        "High": f"High Risk - Strong fraud indicators detected (Score: {score:.2f}%)",
    }[risk_level]
    css_class = {"Low": "alert-low", "Medium": "alert-medium", "High": "alert-high"}[risk_level]
    st.markdown(f'<div class="alert-box {css_class}">{text}</div>', unsafe_allow_html=True)


def validate_batch_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in REQUIRED_COLUMNS if column not in frame.columns]


def render_page_title(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="main-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="subtitle">{subtitle}</div>', unsafe_allow_html=True)
    st.markdown("<hr class='section-line'>", unsafe_allow_html=True)


try:
    MODEL, SCALER, ENCODER, FEATURE_COLUMNS = load_artifacts()
except FileNotFoundError:
    st.error("Model artifacts are missing. Run `python train_model.py` first.")
    st.stop()


with st.sidebar:
    st.markdown('<div class="sidebar-logo">Anomalyze</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Fraud Detection Engine</div>', unsafe_allow_html=True)
    st.markdown("<hr class='section-line'>", unsafe_allow_html=True)
    mode = st.radio(
        "Analysis Mode",
        ["Single Transaction", "Batch Analysis"],
        format_func=lambda value: "Single Transaction" if value == "Single Transaction" else "Batch Analysis",
        label_visibility="collapsed",
    )
    st.markdown("<hr class='section-line'>", unsafe_allow_html=True)
    st.markdown("**Input Profile**")
    st.caption("Real transaction fields only. No hidden fake features.")
    st.markdown("<hr class='section-line'>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="font-size:0.72rem;color:#536987;line-height:1.95">
        <b style="color:#8aa4ca">MODEL</b><br>
        Random Forest<br>
        Hybrid score engine<br><br>
        <b style="color:#8aa4ca">DATASET</b><br>
        20,000 synthetic transactions<br>
        4.32% fraud rate
        </div>
        """,
        unsafe_allow_html=True,
    )


if mode == "Single Transaction":
    render_page_title("Single Transaction Analysis", "Enter transaction details below")

    col1, col2, col3 = st.columns([1.2, 1.2, 0.9])
    with col1:
        amount = st.number_input("Transaction Amount (Rs)", min_value=0.0, value=2500.0, step=100.0)
        merchant_type = st.selectbox("Merchant Type", ["food", "electronics", "travel", "grocery"])
        device_type = st.selectbox("Device Type", ["mobile", "desktop"])
    with col2:
        time_value = st.number_input("Transaction Time (0-23.99 hour)", min_value=0.0, max_value=23.99, value=14.50, step=0.25)
        transaction_type = st.selectbox("Transaction Type", ["online", "POS"])
        location_type = st.selectbox("Location Type", ["same_city", "different_city", "international"])
    with col3:
        transaction_count_24h = st.slider("Transactions in last 24h", min_value=1, max_value=30, value=3)
        avg_amount_24h = st.number_input("Average amount in last 24h", min_value=0.0, value=1800.0, step=100.0)
        is_international = st.checkbox("International", value=False)
        st.markdown("<br>", unsafe_allow_html=True)
        analyze = st.button("Analyze", use_container_width=True)

    if is_international and location_type != "international":
        st.warning("`is_international` is enabled while `location_type` is not `international`. Both values will be used as entered.")

    if analyze:
        input_frame = pd.DataFrame(
            [
                {
                    "amount": amount,
                    "time": time_value,
                    "transaction_count_24h": transaction_count_24h,
                    "avg_amount_24h": avg_amount_24h,
                    "merchant_type": merchant_type,
                    "transaction_type": transaction_type,
                    "device_type": device_type,
                    "location_type": location_type,
                    "is_international": int(is_international),
                }
            ]
        )

        results = score_transactions(input_frame, MODEL, SCALER, ENCODER)
        result = results.iloc[0]
        risk_color = metric_value_color(result["risk_level"])

        render_alert(result["risk_level"], float(result["anomaly_score"]))

        m1, m2, m3 = st.columns(3)
        for column, label, value in [
            (m1, "Anomaly Score", f"{result['anomaly_score']:.2f}%"),
            (m2, "Risk Level", result["risk_level"]),
            (m3, "Prediction", result["prediction"]),
        ]:
            with column:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">{label}</div>
                        <div class="metric-value" style="color:{risk_color};">{value}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### Anomaly Score Gauge")
        st.progress(min(int(result["anomaly_score"]), 100))
        st.caption(f"Score: {result['anomaly_score']:.2f} / 100")
        st.markdown(
            f"**Model Confidence:** <span class='confidence-chip'>{result['confidence']:.2f}%</span>",
            unsafe_allow_html=True,
        )

        details_left, details_right = st.columns(2)
        with details_left:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown("### Decision Summary")
            st.write(f"ML Probability: {result['ml_probability']:.2f}%")
            st.write(f"Rule Score: {result['rule_score']} / 100")
            st.write(f"Merchant Type: {merchant_type}")
            st.write(f"Location Type: {location_type}")
            st.markdown("</div>", unsafe_allow_html=True)

        with details_right:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown("### Explanation Panel")
            for explanation in result["explanations"].split(" | "):
                st.write(f"- {explanation}")
            st.markdown("</div>", unsafe_allow_html=True)

        viz_left, viz_right = st.columns(2)
        with viz_left:
            render_feature_importance(MODEL, FEATURE_COLUMNS)
        with viz_right:
            render_risk_distribution(results)

else:
    render_page_title("Batch Transaction Analysis", "Upload a CSV with the required input columns")

    sample_batch = pd.DataFrame(
        [
            {
                "amount": 320.0,
                "time": 13.25,
                "transaction_count_24h": 2,
                "avg_amount_24h": 280.0,
                "merchant_type": "grocery",
                "transaction_type": "POS",
                "device_type": "mobile",
                "location_type": "same_city",
                "is_international": 0,
            },
            {
                "amount": 22000.0,
                "time": 23.40,
                "transaction_count_24h": 14,
                "avg_amount_24h": 3200.0,
                "merchant_type": "electronics",
                "transaction_type": "online",
                "device_type": "desktop",
                "location_type": "international",
                "is_international": 1,
            },
        ]
    )

    st.download_button(
        "Download Sample CSV",
        data=sample_batch.to_csv(index=False).encode("utf-8"),
        file_name="anomalyze_batch_template.csv",
        mime="text/csv",
    )
    uploaded_file = st.file_uploader("Upload your transactions CSV", type=["csv"])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        missing_columns = validate_batch_columns(batch_df)

        if missing_columns:
            st.error(f"Missing required columns: {', '.join(missing_columns)}")
        else:
            batch_df = batch_df.copy()
            for column in NUMERICAL_FEATURES + BINARY_FEATURES:
                batch_df[column] = pd.to_numeric(batch_df[column], errors="coerce")

            if batch_df[REQUIRED_COLUMNS].isnull().any().any():
                st.error("Uploaded file contains invalid or blank values in required fields.")
            else:
                status_box = st.info(f"Loaded {len(batch_df)} transactions. Processing...")
                results = score_transactions(batch_df[REQUIRED_COLUMNS], MODEL, SCALER, ENCODER)
                status_box.success(f"Processed {len(results)} transactions successfully.")

                st.markdown("## Results Table")
                display_columns = [
                    "amount",
                    "time",
                    "transaction_count_24h",
                    "avg_amount_24h",
                    "merchant_type",
                    "transaction_type",
                    "device_type",
                    "location_type",
                    "is_international",
                    "ml_probability",
                    "rule_score",
                    "anomaly_score",
                    "risk_level",
                    "risk_reason",
                    "prediction",
                    "confidence",
                ]
                st.dataframe(
                    style_predictions(results[display_columns]),
                    width=2200,
                    use_container_width=False,
                    hide_index=True,
                )

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("## Summary Statistics")
                low_count = int((results["risk_level"] == "Low").sum())
                med_count = int((results["risk_level"] == "Medium").sum())
                high_count = int((results["risk_level"] == "High").sum())

                s1, s2, s3, s4 = st.columns(4)
                for column, label, value, color in [
                    (s1, "Total Transactions", len(results), LOW),
                    (s2, "Low Risk", low_count, LOW),
                    (s3, "Medium Risk", med_count, MEDIUM),
                    (s4, "High Risk", high_count, HIGH),
                ]:
                    with column:
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <div class="metric-label">{label}</div>
                                <div class="metric-value" style="color:{color};">{value}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("## Risk Distribution")
                render_risk_distribution(results)

                st.markdown("<div style='height: 0.85rem;'></div>", unsafe_allow_html=True)
                feature_panel, download_panel = st.columns([1.35, 1], gap="large")
                with feature_panel:
                    st.markdown("### Feature Importance")
                    render_feature_importance(MODEL, FEATURE_COLUMNS)
                with download_panel:
                    st.markdown("### Export Results")
                    st.write("Download the full scored batch with ML probability, rule score, risk level, prediction, and explanation text.")
                    st.download_button(
                        "Download Results CSV",
                        data=results.to_csv(index=False).encode("utf-8"),
                        file_name="anomalyze_batch_results.csv",
                        mime="text/csv",
                    )
