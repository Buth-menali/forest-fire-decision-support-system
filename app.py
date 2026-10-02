import json
import os
import pickle

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Forest Fire Decision Support System",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------
# HELPERS
# -----------------------------

def html(content: str):
    """Render HTML cleanly.
    Markdown treats lines indented by 4+ spaces as code blocks and blank
    lines can end an HTML block, so both are removed first."""
    cleaned = "\n".join(
        line.strip() for line in content.splitlines() if line.strip()
    )
    st.markdown(cleaned, unsafe_allow_html=True)


def stretch(widget, *args, **kwargs):
    """Make a widget fill its container on any Streamlit version."""
    try:
        return widget(*args, width="stretch", **kwargs)
    except TypeError:
        return widget(*args, use_container_width=True, **kwargs)


# Accent colours (R, G, B) used for the risk levels
RISK_COLORS = {
    "low": "34, 197, 94",
    "moderate": "234, 179, 8",
    "high": "239, 68, 68",
}


# -----------------------------
# LOAD MODEL
# -----------------------------

if not os.path.exists("fire_model.pkl"):
    st.error("fire_model.pkl not found.")
    st.stop()

with open("fire_model.pkl", "rb") as file:
    model = pickle.load(file)

# Accuracy scores saved by train_model.py (optional file)
metrics = None
if os.path.exists("model_metrics.json"):
    try:
        with open("model_metrics.json") as file:
            metrics = json.load(file)
    except (OSError, ValueError):
        metrics = None


# -----------------------------
# STYLES (work in light and dark mode)
# -----------------------------
# The page background and text colour come from Streamlit's own theme.
# Cards use see-through greys and "inherit" text, so they adapt
# automatically when the theme changes.

html("""
<style>
.hero {
    min-height: 330px;
    border-radius: 24px;
    padding: 56px;
    margin-bottom: 28px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    background-image:
        linear-gradient(100deg, rgba(3, 12, 8, 0.88), rgba(3, 12, 8, 0.45)),
        url("https://images.unsplash.com/photo-1448375240586-882707db888b?auto=format&fit=crop&w=1800&q=85");
    background-size: cover;
    background-position: center;
}

.hero-label {
    align-self: flex-start;
    padding: 7px 14px;
    border-radius: 999px;
    background-color: rgba(255, 255, 255, 0.16);
    color: #e6efe9;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 1.2px;
}

.hero-title {
    font-size: 44px;
    font-weight: 700;
    line-height: 1.12;
    color: #ffffff;
    margin-top: 18px;
}

.hero-text {
    font-size: 17px;
    line-height: 1.6;
    color: #dbe6df;
    margin-top: 16px;
    max-width: 640px;
}

.card {
    background-color: rgba(128, 128, 128, 0.10);
    border: 1px solid rgba(128, 128, 128, 0.28);
    border-radius: 18px;
    padding: 24px;
    margin-bottom: 18px;
    color: inherit;
}

.card-title,
.prediction-title {
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    opacity: 0.65;
}

.card-value {
    font-size: 30px;
    font-weight: 700;
    margin-top: 8px;
    color: inherit;
}

.card-text {
    line-height: 1.7;
    opacity: 0.8;
    margin: 12px 0 0 0;
}

.prediction {
    background-color: rgba(var(--accent), 0.14);
    border: 1px solid rgba(var(--accent), 0.55);
    border-radius: 18px;
    padding: 24px;
    margin-bottom: 18px;
    color: inherit;
}

.prediction-value {
    font-size: 44px;
    font-weight: 800;
    margin-top: 6px;
    color: inherit;
}

.risk-pill {
    display: inline-block;
    padding: 9px 20px;
    border-radius: 999px;
    font-weight: 600;
    color: inherit;
    background-color: rgba(var(--accent), 0.16);
    border: 1px solid rgba(var(--accent), 0.6);
}

.footer {
    text-align: center;
    margin-top: 48px;
    padding: 24px;
    opacity: 0.6;
    border-top: 1px solid rgba(128, 128, 128, 0.30);
}

@media (max-width: 640px) {
    .hero { padding: 32px 24px; }
    .hero-title { font-size: 32px; }
    .card-value { font-size: 24px; }
    .prediction-value { font-size: 36px; }
}
</style>
""")


# -----------------------------
# HERO SECTION
# -----------------------------

html("""
<div class="hero">
    <div class="hero-label">AI-ASSISTED DECISION SUPPORT</div>
    <div class="hero-title">Forest Fire Risk<br>Assessment System</div>
    <div class="hero-text">
        Analyse forest conditions and estimate potential
        fire impact using a machine learning model trained
        on forest-fire simulation data.
    </div>
</div>
""")


# -----------------------------
# SIDEBAR
# -----------------------------

st.sidebar.title("Forest Conditions")
st.sidebar.write("Configure the forest scenario.")
st.sidebar.divider()

density = st.sidebar.slider("Forest Density (%)", 5, 100, 50, 5)

firebreak = st.sidebar.selectbox(
    "Firebreak",
    [0, 1],
    format_func=lambda x: "Present" if x == 1 else "Absent",
)

regrowth_chance = st.sidebar.slider("Regrowth Chance (%)", 0, 100, 1, 1)
regrowth_time = st.sidebar.slider("Regrowth Time (ticks)", 1, 100, 20, 1)
reignite_chance = st.sidebar.slider("Reignite Chance (%)", 0, 100, 5, 1)

st.sidebar.divider()

predict = stretch(st.sidebar.button, "Generate Prediction")


# -----------------------------
# INPUT DATA
# -----------------------------

input_data = pd.DataFrame({
    "density": [density],
    "firebreak?": [firebreak],
    "regrowth-chance": [regrowth_chance],
    "regrowth-time": [regrowth_time],
    "reignite-chance": [reignite_chance],
})

firebreak_text = "Present" if firebreak == 1 else "Absent"


# -----------------------------
# PAGE TITLE
# -----------------------------

st.subheader("Fire Impact Prediction")
st.write(
    "Adjust the environmental conditions and generate "
    "a model-based prediction."
)


# -----------------------------
# INITIAL INFORMATION
# -----------------------------

if not predict:

    col1, col2 = st.columns([1.4, 1])

    with col1:
        html("""
        <div class="card">
            <div class="card-title">System Overview</div>
            <div class="card-value">Configure a forest scenario</div>
            <p class="card-text">
                Select the forest conditions from the control panel
                and generate a machine learning estimate of potential
                burned area.
            </p>
        </div>
        """)

    with col2:
        stretch(
            st.image,
            "https://images.unsplash.com/photo-1448375240586-882707db888b?auto=format&fit=crop&w=900&q=80",
        )


# -----------------------------
# PREDICTION
# -----------------------------

if predict:

    prediction = model.predict(input_data)
    predicted_burned = max(0.0, min(100.0, float(prediction[0])))

    # Risk level
    if predicted_burned < 20:
        risk, level = "🟢 Low", "low"
        message = "The predicted burned area is relatively low."
    elif predicted_burned < 50:
        risk, level = "🟡 Moderate", "moderate"
        message = "The predicted burned area indicates moderate fire impact."
    else:
        risk, level = "🔴 High", "high"
        message = "The predicted burned area indicates high fire impact."

    accent = RISK_COLORS[level]

    # Result cards
    st.subheader("Prediction Result")

    col1, col2, col3 = st.columns(3)

    with col1:
        html(f"""
        <div class="prediction" style="--accent: {accent};">
            <div class="prediction-title">Predicted Burned Area</div>
            <div class="prediction-value">{predicted_burned:.2f}%</div>
        </div>
        """)

    with col2:
        html(f"""
        <div class="card">
            <div class="card-title">Forest Density</div>
            <div class="card-value">{density}%</div>
        </div>
        """)

    with col3:
        html(f"""
        <div class="card">
            <div class="card-title">Firebreak</div>
            <div class="card-value">{firebreak_text}</div>
        </div>
        """)

    # Risk
    st.subheader("Risk Assessment")
    html(f'<div class="risk-pill" style="--accent: {accent};">{risk} Risk</div>')
    st.write(message)

    # Impact
    st.subheader("Predicted Impact")
    st.progress(predicted_burned / 100)
    st.caption(f"Estimated burned area: {predicted_burned:.2f}%")

    # Input summary
    st.subheader("Selected Conditions")

    summary = pd.DataFrame({
        "Parameter": [
            "Forest Density",
            "Firebreak",
            "Regrowth Chance",
            "Regrowth Time",
            "Reignite Chance",
        ],
        "Value": [
            f"{density}%",
            firebreak_text,
            f"{regrowth_chance}%",
            f"{regrowth_time} ticks",
            f"{reignite_chance}%",
        ],
    })

    stretch(st.dataframe, summary, hide_index=True)

    # Model accuracy
    st.subheader("Model Accuracy")

    if metrics:
        m1, m2, m3 = st.columns(3)

        with m1:
            html(f"""
            <div class="card">
                <div class="card-title">R² Score</div>
                <div class="card-value">{metrics["r2"]:.3f}</div>
            </div>
            """)

        with m2:
            html(f"""
            <div class="card">
                <div class="card-title">R² (Unseen Settings)</div>
                <div class="card-value">{metrics["r2_unseen"]:.3f}</div>
            </div>
            """)

        with m3:
            html(f"""
            <div class="card">
                <div class="card-title">Mean Absolute Error</div>
                <div class="card-value">{metrics["mae"]:.2f}%</div>
            </div>
            """)

        st.caption(
            f"Based on {metrics['training_runs']} simulation runs. "
            "R² closer to 1 means the model explains more of the variation "
            "in burned area."
        )
    else:
        st.info("Run train_model.py to generate the model accuracy scores.")

    # Model information
    with st.expander("Model Information"):
        st.write("Model: Random Forest")
        st.write(f"Raw prediction: {prediction[0]:.4f}")


# -----------------------------
# FOOTER
# -----------------------------

html("""
<div class="footer">
    Forest Fire Decision Support System<br><br>
    NetLogo Simulation | Machine Learning | Streamlit
</div>
""")