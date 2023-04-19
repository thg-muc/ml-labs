"""Module to demonstrate a multi-output baseline-model using sklearn."""

# * Author(s): thomas.glanzer@gmail.com
# * Creation : Apr 2023
# * License  : MIT


# %% ############################################
# * Import required Libraries and set Variables

import pandas as pd
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor

SEED = 42

# %% ############################################
# * Sample Data

# Sample data generation (replace this with your actual data)
# ... create data with 4000 rows and 20 features and 80 targets
sample_data = make_regression(n_samples=4000, n_features=20, n_targets=80, random_state=SEED)
# ... create a pandas dataframe from the sample data (features) and add column names
X = pd.DataFrame(sample_data[0], columns=[f"feature_{i}" for i in range(20)])
# ... create a pandas dataframe from the sample data (targets) and add column names
y = pd.DataFrame(sample_data[1], columns=[f"target_{i}" for i in range(80)])

# Scale each column of X and y to the range [0, 10] and convert to integers
for col in X.columns:
    X[col] = (X[col] - X[col].min()) / (X[col].max() - X[col].min()) * 10
    X[col] = X[col].astype(int)
for col in y.columns:
    y[col] = (y[col] - y[col].min()) / (y[col].max() - y[col].min()) * 10
    y[col] = y[col].astype(int)

print(X.head())
print(y.head())

# %% ############################################
# * Baseline Model

# Create a multi-output baseline model with some basic hyperparameters
model = MultiOutputRegressor(RandomForestRegressor(n_estimators=50, max_depth=8,
                                                   random_state=SEED, n_jobs=-1))
# Make a train-test-split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=SEED)

# Train the model (will take 5-20 seconds)
model.fit(X_train, y_train)

# Predict on the test set and show first 5 predictions
y_pred = pd.DataFrame(model.predict(X_test), columns=y.columns)

# Round predictions to the nearest integer and make sure they are in the range [0, 10]
y_pred = (y_pred.round().clip(0, 10)).astype(int)

# Show the first 5 predictions
print(y_pred.head())
print()

# Show some regression scores
print(f'Model R2 Score: {r2_score(y_test, y_pred):.3f}')
print(f'Prediction Mean Squared Error: {mean_squared_error(y_test, y_pred):.3f}')
print(f'Prediction Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.3f}')

# %%
