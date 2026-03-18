from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import torch
import torch.nn as nn
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///weather_users.db'
app.config['SECRET_KEY'] = 'your-secret-key-change-this'
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# User model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)

    def set_password(self, password):
        self.password = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password, password)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Create tables
with app.app_context():
    db.create_all()

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
@login_required
def home():
    return render_template("index.html", username=current_user.username)


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('home'))
        else:
            flash("Invalid username or password", "error")
    
    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    
    if request.method == "POST":
        username = request.form.get("username")
        email = request.form.get("email")
        password = request.form.get("password")
        password_confirm = request.form.get("password_confirm")
        
        if password != password_confirm:
            flash("Passwords do not match", "error")
            return redirect(url_for('register'))
        
        if User.query.filter_by(username=username).first():
            flash("Username already exists", "error")
            return redirect(url_for('register'))
        
        if User.query.filter_by(email=email).first():
            flash("Email already registered", "error")
            return redirect(url_for('register'))
        
        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        
        flash("Account created successfully! Please login.", "success")
        return redirect(url_for('login'))
    
    return render_template("register.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "success")
    return redirect(url_for('login'))


@app.route("/predict", methods=["POST"])
@login_required
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
            "result.html",
            prediction_text=f"Predicted Weather: {weather_label}",
            **form
        )

    except Exception as e:
        print("ERROR:", e)
        return render_template(
            "result.html",
            prediction_text="Error occurred. Check inputs.",
            temperature="N/A",
            humidity="N/A",
            wind_speed="N/A",
            precipitation="N/A",
            cloud_cover="N/A",
            uv_index="N/A",
            season="N/A",
            visibility="N/A",
            location="N/A",
            pressure="N/A"
        )


if __name__ == "__main__":
    app.run(debug=True)