# ==============================
# Medical Insurance Cost Prediction
# Regression Project
# ==============================

# 1. Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 2. Load the dataset
data = pd.read_csv("data/insurance.csv")

# Display basic information
print("First 5 rows of the dataset:")
print(data.head())
print("\nDataset Information:")
print(data.info())


# 3. Encode categorical variables
# Convert text data into numerical format
data['sex'] = data['sex'].map({'male': 0, 'female': 1})
data['smoker'] = data['smoker'].map({'no': 0, 'yes': 1})

# One-hot encode 'region'
data = pd.get_dummies(data, columns=['region'], drop_first=True)


# 4. Separate features (X) and target (y)
X = data.drop('charges', axis=1)
y = data['charges']


# 5. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 6. Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)


# 7. Make predictions on test data
y_pred = model.predict(X_test)


# 8. Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")


# 9. Visualize Actual vs Predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Insurance Charges")
plt.ylabel("Predicted Insurance Charges")
plt.title("Actual vs Predicted Medical Insurance Charges")
plt.show()
