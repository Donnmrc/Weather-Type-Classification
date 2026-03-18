# train_dl_model.py
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# -----------------------------
# 1️⃣ Load dataset
# -----------------------------
data = pd.read_csv("weather_data.csv")

# -----------------------------
# 2️⃣ Encode categorical columns
# -----------------------------
label_encoders = {}

for column in data.columns:
    if data[column].dtype == "object":
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

# Save encoders
joblib.dump(label_encoders, "encoders.pkl")

# -----------------------------
# 3️⃣ Split features and target
# -----------------------------
X = data.drop("Weather Type", axis=1)
y = data["Weather Type"]

# -----------------------------
# 4️⃣ Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 5️⃣ Scale features
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

joblib.dump(scaler, "scaler.pkl")

# -----------------------------
# 6️⃣ Convert to PyTorch tensors
# -----------------------------
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.long)
y_test = torch.tensor(y_test.values, dtype=torch.long)

# -----------------------------
# 7️⃣ Define Neural Network
# -----------------------------
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

model = WeatherNN(X_train.shape[1])

# -----------------------------
# 8️⃣ Loss + optimizer
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -----------------------------
# 9️⃣ Training loop
# -----------------------------
print("Starting training...")
epochs = 50

for epoch in range(epochs):

    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}/{epochs}  Loss: {loss.item():.4f}")

print("✅ TRAINING COMPLETE")

# -----------------------------
# 🔟 Save model
# -----------------------------
torch.save(model.state_dict(), "weather_model.pth")
print("Model saved as weather_model.pth")

# -----------------------------
# 11️⃣ Evaluate accuracy
# -----------------------------
with torch.no_grad():
    test_outputs = model(X_test)
    _, predicted = torch.max(test_outputs, 1)
    accuracy = (predicted == y_test).float().mean()

print(f"Test Accuracy: {accuracy:.4f}")