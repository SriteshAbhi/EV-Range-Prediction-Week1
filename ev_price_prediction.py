import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
try:
    df = pd.read_csv("C:/Users/ramag/Week-1/electric_vehicle_data.csv")
    print("✅ Data loaded successfully!")
except FileNotFoundError:
    print("❌ File not found! Please check the file path.")
    exit()

# Show small preview
print("\nSample Data:\n", df.head())
print("\nColumns in dataset:\n", df.columns.tolist())

# We'll predict 'range_km' instead of 'price'
X = df[['battery_capacity_kWh', 'top_speed_kmh', 'efficiency_wh_per_km']].copy()
y = df['range_km']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("\n📊 Model Evaluation:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Display few predictions
result = pd.DataFrame({
    'Actual Range (km)': y_test.values,
    'Predicted Range (km)': y_pred
})
print("\n🔍 Comparison (Actual vs Predicted):\n", result.head())
import matplotlib.pyplot as plt

# Scatter Plot: Actual vs Predicted
plt.figure(figsize=(7,5))
plt.scatter(y_test, y_pred, color='royalblue', edgecolor='black', alpha=0.7)
plt.title("Actual vs Predicted Electric Vehicle Range")
plt.xlabel("Actual Range (km)")
plt.ylabel("Predicted Range (km)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("Range_Prediction_Comparison.png")
plt.show()
