from pathlib import Path
import json
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.simulator import FEATURES, save_sensor_data
from src.train import train_and_evaluate
from src.predict import predict_machine, MODEL_PATH

DATA_PATH = Path("data/raw/virtual_sensor_data.csv")
st.set_page_config(page_title="AI Predictive Maintenance", page_icon="🛠️", layout="wide")

st.title("🛠️ AI-Powered Predictive Maintenance")
st.caption("Virtual IoT sensor simulation • Machine-learning demonstration • No physical hardware required")

with st.sidebar:
    st.header("⚙️ Project Controls")
    st.write("Create simulated sensor data and train the model.")
    sample_count = st.select_slider("Number of virtual records", options=[1000, 2500, 5000, 10000, 20000], value=10000)
    seed = st.number_input("Random seed", min_value=0, max_value=999999, value=42, step=1)
    if st.button("Generate data + train model", type="primary", use_container_width=True):
        with st.spinner("Generating data and training baseline models..."):
            data = save_sensor_data(n_samples=int(sample_count), seed=int(seed))
            result = train_and_evaluate(data)
        st.session_state["training_result"] = result
        st.success(f"Training complete. Best model: {result['best_model']}")
    st.divider()
    st.caption("All readings and labels are simulated for education. This is not a certified safety system.")

data = None
if DATA_PATH.exists():
    try:
        data = pd.read_csv(DATA_PATH)
    except Exception as exc:
        st.error(f"Could not read dataset: {exc}")

if not MODEL_PATH.exists():
    st.info("Welcome! Use **Generate data + train model** in the sidebar to prepare the dataset and train the model. Then use the controls below to score a virtual machine reading.")
else:
    st.success("A trained model is available.")

st.subheader("🔧 Virtual Machine Sensor Inputs")
default_values = {
    "temperature": 64.0, "vibration": 2.2, "current": 8.5,
    "rotational_speed": 1500.0, "torque": 40.0, "tool_wear": 75.0
}
labels = {
    "temperature": ("Temperature (°C)", 20.0, 120.0, 1.0),
    "vibration": ("Vibration (mm/s)", 0.1, 15.0, 0.1),
    "current": ("Current (A)", 1.0, 22.0, 0.1),
    "rotational_speed": ("Rotational speed (RPM)", 500.0, 2300.0, 10.0),
    "torque": ("Torque (Nm)", 5.0, 80.0, 1.0),
    "tool_wear": ("Tool wear (minutes)", 0.0, 360.0, 1.0),
}
cols = st.columns(3)
reading = {}
for idx, feature in enumerate(FEATURES):
    label, low, high, step = labels[feature]
    with cols[idx % 3]:
        reading[feature] = st.slider(
            label, min_value=float(low), max_value=float(high),
            value=float(default_values[feature]), step=float(step), key=f"sensor_{feature}"
        )

predict_col, reset_col = st.columns([1, 4])
with predict_col:
    run_prediction = st.button("🔍 Predict condition", type="primary", use_container_width=True)
if run_prediction:
    if not MODEL_PATH.exists():
        st.warning("Train the model first using the sidebar button.")
    else:
        history = data[FEATURES].tail(4) if data is not None and len(data) else None
        try:
            result = predict_machine(reading, history=history)
            st.session_state["last_prediction"] = result
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

result = st.session_state.get("last_prediction")
if result:
    st.subheader("📊 Prediction Result")
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted failure", "YES" if result["predicted_failure"] else "NO")
    c2.metric("Failure probability", f"{result['failure_probability']:.1%}")
    c3.metric("Model", result["model_name"])
    if result["alert"] == "HIGH RISK":
        st.error("🚨 HIGH RISK — inspect the simulated readings. This is a demo alert, not an operational safety instruction.")
    elif result["alert"] == "WATCH":
        st.warning("⚠️ WATCH — the model estimates elevated risk; review the readings.")
    else:
        st.success("✅ NORMAL — no high-risk alert from this demo model.")

st.divider()
left, right = st.columns(2)
with left:
    st.subheader("📈 Virtual Sensor Trends")
    if data is not None and not data.empty:
        trend_feature = st.selectbox("Choose sensor", FEATURES, format_func=lambda x: labels[x][0])
        fig, ax = plt.subplots(figsize=(8, 3.4))
        sample = data.tail(200)
        ax.plot(range(len(sample)), sample[trend_feature], linewidth=1.5)
        ax.set_xlabel("Recent simulated records")
        ax.set_ylabel(labels[trend_feature][0])
        ax.grid(alpha=0.25)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.caption("Generate data to view sensor trends.")
with right:
    st.subheader("🧾 Failure Label Distribution")
    if data is not None and "failure" in data.columns:
        counts = data["failure"].value_counts().reindex([0, 1], fill_value=0)
        fig, ax = plt.subplots(figsize=(8, 3.4))
        ax.bar(["Normal (0)", "Failure (1)"], counts.values)
        ax.set_ylabel("Records")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Labels are generated by predefined thresholds in the simulator.")
    else:
        st.caption("Generate data to view the label distribution.")

st.subheader("🗂️ Sensor Dataset Preview")
if data is not None:
    st.write(f"**{len(data):,} records** · **{int(data['failure'].sum()):,} simulated failure labels**")
    st.dataframe(data.tail(20), use_container_width=True)
    st.download_button(
        "Download virtual sensor CSV", data=data.to_csv(index=False).encode("utf-8"),
        file_name="virtual_sensor_data.csv", mime="text/csv"
    )
else:
    st.caption("The dataset will appear here after generation.")

with st.expander("📋 Model evaluation"):
    result_train = st.session_state.get("training_result")
    metrics_path = Path("outputs/metrics/model_metrics.json")
    if result_train:
        st.write(f"**Selected model:** {result_train['best_model']}")
        st.dataframe(pd.DataFrame(result_train["metrics"]).T, use_container_width=True)
    elif metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        st.write(f"**Selected model:** {metrics['best_model']}")
        st.dataframe(pd.DataFrame(metrics["models"]).T, use_container_width=True)
    else:
        st.caption("Train the model to see accuracy, precision, recall, F1, and ROC-AUC.")

st.caption("Educational prototype only. The synthetic labels are rule-generated; reported scores do not establish real-world predictive maintenance accuracy. Do not connect this demo to machinery or use it for safety-critical decisions.")
