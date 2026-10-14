# Architecture

1. **Virtual data generation:** `src/simulator.py` creates synthetic timestamped sensor readings and threshold-based demonstration labels.
2. **Feature engineering:** `src/train.py` adds rolling means and standard deviations for temperature, vibration, and current.
3. **Training:** Logistic Regression and Random Forest are compared on a stratified hold-out split.
4. **Evaluation:** Accuracy, precision, recall, F1-score, ROC-AUC, a confusion matrix, and a model-comparison chart are saved.
5. **Inference:** `src/predict.py` loads the selected model bundle and scores a single virtual sensor reading.
6. **User interface:** `app.py` exposes sensor sliders, training controls, charts, metrics, and CSV download through Streamlit.

The pipeline is a teaching demo, not a validated industrial deployment.
