"""Command-line entry point: create virtual data and train models."""
from src.simulator import save_sensor_data
from src.train import train_and_evaluate

if __name__ == "__main__":
    data = save_sensor_data()
    print(f"Created {len(data):,} virtual sensor records.")
    result = train_and_evaluate(data)
    print(f"Best model: {result['best_model']}")
    print("Run the dashboard with: streamlit run app.py")
