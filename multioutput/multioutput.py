"""Module to demonstrate a multi-output baseline-model using sklearn."""

# * Author(s): thomas.glanzer@gmail.com
# * Creation : Apr 2023
# * License  : MIT


# %% ############################################
# * Import required Libraries and set Variables

import pandas as pd
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

SEED = 42

# %% ############################################
# * Sample Data

# Sample data generation (replace this with your actual data)
# ... create data with 4000 rows and 20 features and 80 targets
sample_data = make_regression(
    n_samples=4000, n_features=20, n_targets=80, random_state=SEED)
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

# Prepare a train-test-split for the baseline models
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=SEED)


def finalize_pred(x):
    """Round preds to the nearest int and have them in range [0, 10]."""
    return (x.round().clip(0, 10)).astype(int)


# %% ############################################
# * Baseline Model (Random Forest)

# Create a multi-output RF baseline model
# ... (MultiOutputRegressor is not strictly necessary, but can increase performance)
rfr = MultiOutputRegressor(RandomForestRegressor(n_estimators=50, max_depth=8,
                                                 random_state=SEED, n_jobs=-1))

# Train the model (will take 5-15 seconds)
rfr.fit(X_train, y_train)

# Predict on the test set and show first 5 predictions
y_pred = finalize_pred(pd.DataFrame(rfr.predict(X_test), columns=y.columns))

# Show some regression scores
print('\nBaseline Model (Multi Output - Random Forest)')
print(f'Model R2 Score: {r2_score(y_test, y_pred):.3f}')
print(f'Prediction Mean Squared Error: {mean_squared_error(y_test, y_pred):.3f}')
print(f'Prediction Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.3f}')

# %% ############################################
# * Baseline Model (Linear Regression)

# Create a preprocessing step to scale all features to the range [0, 1]
# ... (MultiOutputRegressor is not strictly necessary, but can increase performance)
scaler = MinMaxScaler()

# Create a Linear Regression baseline model
reg = MultiOutputRegressor(LinearRegression(n_jobs=-1))

# Create a pipeline to chain the preprocessing step and the model
reg = make_pipeline(scaler, reg)

# Train the model (will take 5-15 seconds)
reg.fit(X_train, y_train)

# Predict on the test set and show first 5 predictions
y_pred = finalize_pred(pd.DataFrame(reg.predict(X_test), columns=y.columns))

# Show some regression scores
print('\nBaseline Model (Multi Output - Linear Regression)')
print(f'Model R2 Score: {r2_score(y_test, y_pred):.3f}')
print(f'Prediction Mean Squared Error: {mean_squared_error(y_test, y_pred):.3f}')
print(f'Prediction Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.3f}')

# %% ############################################
# * Baseline Model (Neural Network / MLP)

# Create a preprocessing step to scale all features to the range [0, 1]
scaler = MinMaxScaler()

# Create a multi-output MLP baseline model
mlp = MLPRegressor(hidden_layer_sizes=(64, 64), activation='relu', solver='adam',
                   max_iter=500, random_state=SEED, learning_rate='adaptive',
                   learning_rate_init=0.01)

# Create a pipeline to chain the preprocessing step and the model
nn_pipe = make_pipeline(scaler, mlp)

# Train the model (will take 10-30 seconds)
nn_pipe.fit(X_train, y_train)

# Predict on the test set and show first 5 predictions
y_pred = finalize_pred(pd.DataFrame(nn_pipe.predict(X_test), columns=y.columns))

# Show some regression scores
print('\nBaseline Model (Neural Network - MLP)')
print(f'Model R2 Score: {r2_score(y_test, y_pred):.3f}')
print(f'Prediction Mean Squared Error: {mean_squared_error(y_test, y_pred):.3f}')
print(f'Prediction Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.3f}')

# %%
