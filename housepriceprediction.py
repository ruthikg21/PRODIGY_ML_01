
import pandas as pd

try:
    df = pd.read_csv('train.csv')
    print(df.shape)
    display(df.head())
except FileNotFoundError:
    print("Error: 'train.csv' not found.")
    df = None  # Set df to None to indicate failure
except pd.errors.ParserError:
    print("Error: Could not parse 'train.csv'. Check file format.")
    df = None
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    df = None

# Examine the shape of the DataFrame.
print("Shape of the DataFrame:", df.shape)

# Display the first few rows of the DataFrame.
display(df.head())

# Get a summary of the DataFrame's structure.
display(df.info())

# Calculate descriptive statistics for numerical features.
display(df.describe())

# Identify and count missing values in each column.
missing_values = df.isnull().sum()
print("\nMissing Values per column:\n", missing_values)

# Analyze the distribution of numerical features using histograms.
import matplotlib.pyplot as plt
df.hist(figsize=(20,20), bins=50)
plt.tight_layout()
plt.show()


# Analyze the distribution of categorical features.
for column in df.select_dtypes(include=['object']).columns:
    print(f"\nValue counts for '{column}':")
    display(df[column].value_counts())


# Impute missing values for numerical features
for col in ['LotFrontage', 'MasVnrArea', 'GarageYrBlt']:
    if df[col].isnull().any():
        if col == 'LotFrontage':
            df[col] = df[col].fillna(df[col].median())  # Median is less sensitive to outliers
        elif col == 'MasVnrArea':
            df[col] = df[col].fillna(0)  # 0 is a reasonable assumption for missing veneer area
        elif col == 'GarageYrBlt':
            df[col] = df[col].fillna(df['YearBuilt']) #Use yearbuilt as imputation for garage year built

# Check for inconsistent data types and correct them if necessary.
# (No inconsistencies found during data exploration, so no action needed here)

# Remove duplicate rows
df.drop_duplicates(inplace=True)

import pandas as pd
from sklearn.preprocessing import StandardScaler

# Identify categorical columns
categorical_cols = df.select_dtypes(include='object').columns

# One-hot encode categorical features
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Identify numerical columns (excluding 'SalePrice')
numerical_cols = df.select_dtypes(exclude='object').columns
numerical_cols = numerical_cols.drop('SalePrice')

# Scale numerical features
scaler = StandardScaler()
df_encoded[numerical_cols] = scaler.fit_transform(df_encoded[numerical_cols])

# The final DataFrame is ready for model training
display(df_encoded.head())

from sklearn.model_selection import train_test_split

# Assuming 'SalePrice' is the target variable
X = df_encoded.drop('SalePrice', axis=1)
y = df_encoded['SalePrice']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display shapes of the resulting DataFrames to verify the split
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

from sklearn.ensemble import RandomForestRegressor

# Initialize the model with hyperparameters
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
try:
    model.fit(X_train, y_train)
    print("Model trained successfully.")
except ValueError as e:
    print(f"Error during model training: {e}")
except Exception as e:
    print(f"An unexpected error occurred during model training: {e}")


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print the metrics
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2: {r2:.4f}")
