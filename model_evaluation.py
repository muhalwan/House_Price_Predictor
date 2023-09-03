# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the California housing dataset
california_data = fetch_california_housing()
df_california = pd.DataFrame(california_data.data, columns=california_data.feature_names)
df_california['MedHouseVal'] = california_data.target

# Standardize the features
X = df_california.drop('MedHouseVal', axis=1)
y = df_california['MedHouseVal']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train a Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred = lr_model.predict(X_test)

# Train a Random Forest Regressor model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Evaluation metrics for Linear Regression
mse_lr = mean_squared_error(y_test, y_pred)
rmse_lr = mean_squared_error(y_test, y_pred, squared=False)
mae_lr = mean_absolute_error(y_test, y_pred)
r2_lr = r2_score(y_test, y_pred)

# Evaluation metrics for Random Forest Regressor
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = mean_squared_error(y_test, y_pred_rf, squared=False)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# Print the results
print("Linear Regression Metrics:")
print(f"MSE: {mse_lr}, RMSE: {rmse_lr}, MAE: {mae_lr}, R2: {r2_lr}")
print("\nRandom Forest Regressor Metrics:")
print(f"MSE: {mse_rf}, RMSE: {rmse_rf}, MAE: {mae_rf}, R2: {r2_rf}")

# Visualization of Predicted vs. Actual values for Random Forest model
plt.scatter(y_test, y_pred_rf, color='blue', alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # y=x line
plt.xlabel('Actual Median House Value')
plt.ylabel('Predicted Median House Value')
plt.title('Actual Median House Value vs. Predicted Median House Value (Random Forest)')
plt.grid(True)
plt.show()
