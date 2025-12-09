import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ------------------------------------------------
# 1) Generate synthetic chiller dataset
# ------------------------------------------------

def generate_synthetic_chiller_data(n=1500):
    np.random.seed(42)

    evap_temp = np.random.uniform(2, 12, n)              # °C
    cond_pressure = np.random.uniform(1, 8, n)           # bar
    flow_rate = np.random.uniform(50, 350, n)            # m³/h
    compressor_current = np.random.uniform(20, 250, n)   # A
    refrigerant_level = np.random.uniform(40, 100, n)    # %

    # Simple fault score rule
    fault_score = (
        (evap_temp > 9) +
        (cond_pressure > 6) +
        (flow_rate < 120) +
        (compressor_current > 210) +
        (refrigerant_level < 55)
    )

    condition = np.where(fault_score >= 3, "Critical",
                  np.where(fault_score == 2, "Degraded", "Healthy"))

    return pd.DataFrame({
        "evap_temp_C": evap_temp,
        "cond_pressure_bar": cond_pressure,
        "flow_rate_m3h": flow_rate,
        "compressor_current_A": compressor_current,
        "refrigerant_level_pct": refrigerant_level,
        "condition": condition,
    })

# ------------------------------------------------
# 2) Cache dataset and train model
# ------------------------------------------------

@st.cache_data
def get_training_data():
    return generate_synthetic_chiller_data()

@st.cache_resource
def train_model(df: pd.DataFrame):
    X = df.drop(columns=["condition"])
    y = df["condition"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42
    )
    model.fit(X_train, y_train)

    return model, X_train

# ------------------------------------------------
# 3) 3D surface plotting function
# ------------------------------------------------

def plot_3d_fault_surface(model, X, feature_x="cond_pressure_bar", feature_y="evap_temp_C"):
    # Range of values for the 2 axes
    x_vals = np.linspace(X[feature_x].min(), X[feature_x].max(), 30)
    y_vals = np.linspace(X[feature_y].min(), X[feature_y].max(), 30)

    xx, yy = np.meshgrid(x_vals, y_vals)

    # Start grid with just the two features we are varying
    grid = pd.DataFrame({
        feature_x: xx.ravel(),
        feature_y: yy.ravel(),
    })

    # For the *other* features, use median values
    for col in X.columns:
        if col not in (feature_x, feature_y):
            grid[col] = X[col].median()

    # 🔴 IMPORTANT: make sure columns are in EXACT same order as during training
    grid = grid[X.columns]

    # Use probability of "Critical" as z-value (0 to 1)
    classes_list = list(model.classes_)
    if "Critical" in classes_list:
        crit_idx = classes_list.index("Critical")
        proba = model.predict_proba(grid)[:, crit_idx]
    else:
        # Fallback: map text labels to numbers
        label_map = {"Healthy": 0, "Degraded": 1, "Critical": 2}
        preds = model.predict(grid)
        proba = np.array([label_map.get(p, 0) for p in preds], dtype=float)

    zz = proba.reshape(xx.shape)

    fig = go.Figure(data=[go.Surface(
        x=xx,
        y=yy,
        z=zz,
        colorscale="Viridis",
        showscale=True
    )])

    fig.update_layout(
        title="3D Surface – Probability of Critical Fault",
        scene=dict(
            xaxis_title=feature_x,
            yaxis_title=feature_y,
            zaxis_title="P(Critical fault)",
        )
    )

    return fig

# ------------------------------------------------
# 4) Streamlit UI
# ------------------------------------------------

st.title("❄️ Chiller Predictive Maintenance Dashboard")

df = get_training_data()
model, X_train = train_model(df)

st.sidebar.header("Input Chiller Measurements")

evap_temp = st.sidebar.slider("Evaporator outlet temperature (°C)", 2.0, 12.0, 6.0)
cond_pressure = st.sidebar.slider("Condenser pressure (bar)", 1.0, 8.0, 3.0)
flow_rate = st.sidebar.slider("Chilled water flow rate (m³/h)", 50.0, 350.0, 180.0)
compressor_current = st.sidebar.slider("Compressor current (A)", 20.0, 250.0, 180.0)
refrigerant_level = st.sidebar.slider("Refrigerant level (%)", 40.0, 100.0, 85.0)

input_df = pd.DataFrame({
    "evap_temp_C": [evap_temp],
    "cond_pressure_bar": [cond_pressure],
    "flow_rate_m3h": [flow_rate],
    "compressor_current_A": [compressor_current],
    "refrigerant_level_pct": [refrigerant_level]
})

prediction = model.predict(input_df)[0]

st.subheader(f"🔍 Predicted Condition: **{prediction}**")

if prediction == "Healthy":
    st.success("✅ System operating normally.")
elif prediction == "Degraded":
    st.warning("⚠️ Efficiency loss detected — inspect soon.")
else:
    st.error("❌ Critical fault — immediate maintenance required!")

# ------------------------------------------------
# 5) Show 3D fault surface
# ------------------------------------------------

st.subheader("🌋 3D Fault Surface View")
fig_3d = plot_3d_fault_surface(model, X_train)
st.plotly_chart(fig_3d, use_container_width=True)
