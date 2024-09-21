# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 20:03:25 2024

@author: Varsha
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = pd.read_csv(r'E:\6th sem\Data Science\DS assignment\simplecardetails.csv')

# Define the features and the target variable
X = data[["Price in 2020"]] 
y = data["price in 2022"]  

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Calculate MAPE (Mean Absolute Percentage Error)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

# Calculate accuracy as (100 - MAPE)
accuracy = 100 - mape

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R² Score: {r2}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"Accuracy of linear regression: {accuracy:.2f}%")

# Create a DataFrame to compare actual and predicted values
comparison_df = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})

# Display the DataFrame
print("\nComparison of Actual and Predicted Values:")
print(comparison_df)