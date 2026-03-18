from flask import Flask, render_template, request
import torch
import torch.nn as nn
import pandas as pd
import joblib

app = Flask(__name__)

# Load scaler + encoders
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")

# Load dataset to get EXACT feature order
data = pd.read_csv("weather_data.csv")

# Remove target column (assumed last column)
feature_columns = data.columns[:-1]

# Model
class WeatherNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 4)
        )

    def forward(self, x):
        return self.model(x)

model = WeatherNN(len(feature_columns))
model.load_state_dict(torch.load("weather_model.pth"))
model.eval()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        form = request.form.to_dict()

        # Build dictionary EXACTLY matching dataset column names
        input_dict = {
            "Temperature": float(form["temperature"]),
            "Humidity": float(form["humidity"]),
            "Wind Speed": float(form["wind_speed"]),
            "Precipitation (%)": float(form["precipitation"]),
            "Cloud Cover": float(form["cloud_cover"]),
            "UV Index": float(form["uv_index"]),
            "Season": form["season"],
            "Visibility (km)": float(form["visibility"]),
            "Location": float(form["location"]),
            "Atmospheric Pressure": float(form["pressure"])
        }

        # Create DataFrame
        df = pd.DataFrame([input_dict])

        # 🔥 CRITICAL FIX: FORCE CORRECT ORDER
        df = df[feature_columns]

        # Encode only Season
        df["Season"] = encoders["Season"].transform(df["Season"])

        df = df.astype(float)

        # Scale
        scaled = scaler.transform(df)

        input_tensor = torch.tensor(scaled, dtype=torch.float32)

        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)

        prediction = predicted.item()

        weather_label = encoders["Weather Type"].inverse_transform([prediction])[0]

        return render_template(
            "index.html",
            prediction_text=f"Predicted Weather: {weather_label}",
            **form
        )

    except Exception as e:
        print("ERROR:", e)
        return render_template(
            "index.html",
            prediction_text="Error occurred. Check inputs."
        )


if __name__ == "__main__":
    app.run(debug=True)