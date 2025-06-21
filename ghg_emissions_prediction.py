# GHG Emissions Prediction Script
# ----------------------------------
# This script reads emission factor data from Excel, trains a Random Forest model to predict emissions,
# and saves the trained model. It supports data from 2010 to 2016.

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Path to your Excel file (MAKE SURE IT'S IN THE SAME FOLDER AS THIS .py FILE)
excel_file = 'SupplyChainEmissionFactorsforUSIndustriesCommodities.xlsx'

# Years of data to process
years = range(2010, 2017)

# Placeholder for all data across years
all_data = []

# Read and combine data from multiple years and sheets
for year in years:
    try:
        df_com = pd.read_excel(excel_file, sheet_name=f'{year}_Detail_Commodity')
        df_ind = pd.read_excel(excel_file, sheet_name=f'{year}_Detail_Industry')
        
        df_com['Source'] = 'Commodity'
        df_ind['Source'] = 'Industry'
        df_com['Year'] = df_ind['Year'] = year
        
        all_data.append(df_com)
        all_data.append(df_ind)
    except Exception as e:
        print(f"Error reading data for year {year}: {e}")

# Combine all yearly data into one DataFrame
data = pd.concat(all_data, ignore_index=True)

# Drop missing values
data.dropna(inplace=True)

# Select numeric features only for modeling
numeric_data = data.select_dtypes(include=[np.number])

# Define features (X) and target (y)
X = numeric_data.drop(columns=['Total Emissions'], errors='ignore')
y = numeric_data['Total Emissions'] if 'Total Emissions' in numeric_data else numeric_data.iloc[:, -1]

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Random Forest Regressor with grid search
model = RandomForestRegressor(random_state=42)
params = {'n_estimators': [50, 100], 'max_depth': [None, 10, 20]}
grid = GridSearchCV(model, params, cv=3, scoring='neg_mean_squared_error')
grid.fit(X_train_scaled, y_train)

# Get best model
best_model = grid.best_estimator_

# Predict on test data
y_pred = best_model.predict(X_test_scaled)

# Evaluate model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f'RMSE: {rmse:.2f}')
print(f'R^2 Score: {r2:.2f}')

# Plot predictions vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel('Actual Emissions')
plt.ylabel('Predicted Emissions')
plt.title('Actual vs Predicted GHG Emissions')
plt.grid(True)
plt.show()

# Save the trained model
joblib.dump(best_model, 'ghg_rf_model.pkl')
print("Model saved as 'ghg_rf_model.pkl'")
