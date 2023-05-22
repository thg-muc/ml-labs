"""Module to demonstrate a multi-output baseline-model using sklearn."""

# * Author(s): thomas.glanzer@gmail.com
# * Creation : Apr 2023
# * License  : MIT


# %% ############################################
# * Import required Libraries and set Variables

import os

import numpy as np
import pandas as pd
import seaborn as sns
from keras.layers import BatchNormalization, Dense, Dropout
from keras.models import Sequential
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from scipy.spatial.distance import cdist
from sklearn.cluster import BisectingKMeans, FeatureAgglomeration, KMeans
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.legacy import Adam

SEED = 42
NR_TOP_FEATURES = 30

# Set the random seed
np.random.seed(SEED)

# %% ############################################
# * Sample Data

# Sample load sample data
df_raw = pd.read_csv(f'sample_data{os.sep}responses.csv')

# Keep only the numeric columns
df = df_raw.select_dtypes(include=[int, float]).dropna(axis=0)

# Keep only numeric columns with range between 1 and 5
df = df.loc[:, (df.min() >= 1) & (df.max() <= 5)]

# Drop records that have NA value
df = df.dropna(axis=0)

# Generate train test split and return indexes
train_idx, test_idx = train_test_split(df.index, random_state=SEED)


def finalize_pred(x, lower, upper):
    """Round preds to the nearest int and have them in range [lower, upper]."""
    return (x.round().clip(lower, upper)).astype(int)

# %% ############################################
# * Baseline Models - Random Column selection


# Randomly select n columns of the dataframe
feature_cols = df.sample(n=NR_TOP_FEATURES, axis=1, random_state=SEED).columns
target_cols = df.drop((feature_cols), axis=1).columns
# Create X and y
X_train = df.loc[train_idx, feature_cols]
X_test = df.loc[test_idx, feature_cols]
y_train = df.loc[train_idx, target_cols]
y_test = df.loc[test_idx, target_cols]

# * Baseline Model (Predict mean of each column)

# Make a dummy prediction
y_pred = DummyRegressor(strategy='median').fit(X_train, y_train).predict(X_test)

# Show some scores
print('\nBaseline Model (Dummy - Median)')
print(f'Model R2 Score: {r2_score(y_test, y_pred):.3f}')
print(f'Prediction Mean Squared Error: {mean_squared_error(y_test, y_pred):.3f}')
print(f'Prediction Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.3f}')

# * Baseline Model (Predict constant = 3 for each column)

# Make a constant prediction (constant = 3)
y_pred = 3 * np.ones(y_test.shape)

# Show some scores
print('\nBaseline Model (Dummy - Constant 3)')
print(f'Model R2 Score: {r2_score(y_test, y_pred):.3f}')
print(f'Prediction Mean Squared Error: {mean_squared_error(y_test, y_pred):.3f}')
print(f'Prediction Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.3f}')

# * Baseline Model (Random Forest)

# Create a multi-output RF baseline model
# ... (MultiOutputRegressor is not strictly necessary, but can increase performance)
rfr = MultiOutputRegressor(RandomForestRegressor(n_estimators=50, max_depth=8,
                                                 random_state=SEED, n_jobs=-1))

# Train the model (will take 5-15 seconds)
rfr.fit(X_train, y_train)

# Make predictions on the test set
y_pred = finalize_pred(pd.DataFrame(rfr.predict(X_test), columns=y_train.columns), 1, 5)

# Show some scores
print('\nBaseline Model (Multi Output - Random Forest)')
print(f'Model R2 Score: {r2_score(y_test, y_pred):.3f}')
print(f'Prediction Mean Squared Error: {mean_squared_error(y_test, y_pred):.3f}')
print(f'Prediction Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.3f}')

# * Baseline Model (Neural Network / MLP)


# Create a multi-output MLP baseline model
mlp = MLPRegressor(hidden_layer_sizes=(256, 64, 16), alpha=0.01, activation='relu',
                   solver='adam', max_iter=1000, learning_rate='adaptive',
                   validation_fraction=0.15, n_iter_no_change=50, batch_size=64,
                   learning_rate_init=0.03, random_state=SEED)

# Create a pipeline to chain the preprocessing step and the model
nn_pipe = make_pipeline(MinMaxScaler(), mlp)

# Train the model (will take 10-25 seconds)
nn_pipe.fit(X_train, y_train)

# Predict on the test set and show first 5 predictions
y_pred = finalize_pred(pd.DataFrame(
    nn_pipe.predict(X_test),
    columns=y_train.columns), 1, 5)

# Show some scores
print('\nBaseline Model (Neural Network - MLP)')
print(f'Model R2 Score: {r2_score(y_test, y_pred):.3f}')
print(f'Prediction Mean Squared Error: {mean_squared_error(y_test, y_pred):.3f}')
print(f'Prediction Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.3f}')


# %% ############################################
# * Perform kmeans

# Randomize the columns of the dataframe
df_cluster = df.T

kmeans = BisectingKMeans(
    n_clusters=NR_TOP_FEATURES,
    n_init=100,
    random_state=SEED,
    algorithm='elkan',
    bisecting_strategy='largest_cluster',
).fit(df_cluster)

# Print cluster labels
for i, label in enumerate(set(kmeans.labels_)):
    features_with_label = [j for j, lab in enumerate(kmeans.labels_) if lab == label]
    print(f'KMEANS cluster {i}: {features_with_label}')


# Get cluster centers
centers = kmeans.cluster_centers_

# Assign each cluster center to the nearest original data point
nearest_points = cdist(centers, df_cluster).argmin(axis=1)


cluster_features = list(df_cluster.iloc[nearest_points].index)


#############


feature_cols = cluster_features
target_cols = df.drop((feature_cols), axis=1).columns
# Create X and y
X_train = df.loc[train_idx, feature_cols]
X_test = df.loc[test_idx, feature_cols]
y_train = df.loc[train_idx, target_cols]
y_test = df.loc[test_idx, target_cols]

# * Baseline Model (Random Forest)

# Create a multi-output RF baseline model
# ... (MultiOutputRegressor is not strictly necessary, but can increase performance)
rfr = MultiOutputRegressor(RandomForestRegressor(n_estimators=50, max_depth=8,
                                                 random_state=SEED, n_jobs=-1))

# Train the model (will take 5-15 seconds)
rfr.fit(X_train, y_train)

# Make predictions on the test set
y_pred = finalize_pred(pd.DataFrame(rfr.predict(X_test), columns=y_train.columns), 1, 5)

# Show some scores
print('\nBaseline Model (Multi Output - Random Forest)')
print(f'Model R2 Score: {r2_score(y_test, y_pred):.3f}')
print(f'Prediction Mean Squared Error: {mean_squared_error(y_test, y_pred):.3f}')
print(f'Prediction Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.3f}')


# * Baseline Model (Neural Network / MLP)

# Create a preprocessing step to scale all features to the range [0, 1]
scaler = MinMaxScaler()

# Create a multi-output MLP baseline model
mlp = MLPRegressor(hidden_layer_sizes=(256, 64, 16), alpha=0.01, activation='relu',
                   solver='adam', max_iter=1000, learning_rate='adaptive',
                   validation_fraction=0.15, n_iter_no_change=50, batch_size=64,
                   learning_rate_init=0.03, random_state=SEED)

# Create a pipeline to chain the preprocessing step and the model
nn_pipe = make_pipeline(scaler, mlp)

# Train the model (will take 10-25 seconds)
nn_pipe.fit(X_train, y_train)

# Predict on the test set and show first 5 predictions
y_pred = finalize_pred(pd.DataFrame(
    nn_pipe.predict(X_test),
    columns=y_train.columns), 1, 5)

# Show some scores
print('\nBaseline Model (Neural Network - MLP)')
print(f'Model R2 Score: {r2_score(y_test, y_pred):.3f}')
print(f'Prediction Mean Squared Error: {mean_squared_error(y_test, y_pred):.3f}')
print(f'Prediction Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.3f}')

# %%
# * Perform clustering


# Randomize the columns of the dataframe
df_cluster = df  # .head(100).T.head(15).T

agglo = FeatureAgglomeration(n_clusters=5, compute_distances=True).fit(df_cluster)

agglo.labels_

for i, label in enumerate(set(agglo.labels_)):
    features_with_label = [j for j, lab in enumerate(agglo.labels_) if lab == label]
    print('Features in agglomeration {}: {}'.format(i, features_with_label))


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # Create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


plt.title("Hierarchical Clustering Dendrogram")
# plot the top three levels of the dendrogram
# plot_dendrogram(agglo, truncate_mode='level', p=6)
# plot_dendrogram(agglo, labels=range(len(df_cluster.columns)), count_sort=True)
plot_dendrogram(agglo, labels=(df_cluster.columns), count_sort=True)
# Rotate the x-axis labels
plt.xticks(rotation=90)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()

# %%
# Autoencoder for feature selection
train_data = (df.loc[train_idx] - 1) / 4
test_data = (df.loc[test_idx] - 1) / 4


# Define the autoencoder model
input_shape = (train_data.shape[1],)
input_layer = Input(shape=input_shape)
encoded = Dense(64, activation='relu')(input_layer)
encoded = Dense(32, activation='relu')(encoded)
decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(train_data.shape[1], activation='sigmoid')(decoded)

autoencoder = Model(input_layer, decoded)

# Compile the model
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Define the early stopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=50,
    restore_best_weights=True)

# Train the model
history = autoencoder.fit(
    train_data,
    train_data,
    epochs=1000,
    batch_size=64,
    validation_data=(
        test_data,
        test_data),
    callbacks=[early_stopping],
    verbose=0)

# Get the weights of the first layer
weights = autoencoder.get_weights()[0]

# Sum the weights for each feature
feature_weights = np.sum(np.abs(weights), axis=1)

# Get the indices of the top  features
top_features = np.argsort(feature_weights)[::-1][:NR_TOP_FEATURES]

# Print the top  features
print(df.columns[top_features])
# Store names of top features
autoencoder_features = df.columns[top_features]


# %%
# * Use auto encoder features for model


feature_cols = autoencoder_features
target_cols = df.drop((feature_cols), axis=1).columns
# Create X and y
X_train = df.loc[train_idx, feature_cols]
X_test = df.loc[test_idx, feature_cols]
y_train = df.loc[train_idx, target_cols]
y_test = df.loc[test_idx, target_cols]

# * Baseline Model (Random Forest)

# Create a multi-output RF baseline model
# ... (MultiOutputRegressor is not strictly necessary, but can increase performance)
rfr = MultiOutputRegressor(RandomForestRegressor(n_estimators=50, max_depth=8,
                                                 random_state=SEED, n_jobs=-1))

# Train the model (will take 5-15 seconds)
rfr.fit(X_train, y_train)

# Make predictions on the test set
y_pred = finalize_pred(pd.DataFrame(rfr.predict(X_test), columns=y_train.columns), 1, 5)

# Show some scores
print('\nBaseline Model (Multi Output - Random Forest)')
print(f'Model R2 Score: {r2_score(y_test, y_pred):.3f}')
print(f'Prediction Mean Squared Error: {mean_squared_error(y_test, y_pred):.3f}')
print(f'Prediction Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.3f}')


# * Baseline Model (Neural Network / MLP)

# Create a preprocessing step to scale all features to the range [0, 1]
scaler = MinMaxScaler()

# Create a multi-output MLP baseline model
mlp = MLPRegressor(hidden_layer_sizes=(256, 64, 16), alpha=0.01, activation='relu',
                   solver='adam', max_iter=1000, learning_rate='adaptive',
                   validation_fraction=0.15, n_iter_no_change=50, batch_size=64,
                   learning_rate_init=0.03, random_state=SEED)

# Create a pipeline to chain the preprocessing step and the model
nn_pipe = make_pipeline(scaler, mlp)

# Train the model (will take 10-25 seconds)
nn_pipe.fit(X_train, y_train)

# Predict on the test set and show first 5 predictions
y_pred = finalize_pred(pd.DataFrame(
    nn_pipe.predict(X_test),
    columns=y_train.columns), 1, 5)

# Show some scores
print('\nBaseline Model (Neural Network - MLP)')
print(f'Model R2 Score: {r2_score(y_test, y_pred):.3f}')
print(f'Prediction Mean Squared Error: {mean_squared_error(y_test, y_pred):.3f}')
print(f'Prediction Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.3f}')


# %%
# * Use auto encoder to predict

# TODO


# %% use correlation analysis


# Generate correlation matrix
corr_matrix = df.corr().abs()

# If Column and Row name are the same, replace 1 with 0
nrows, ncols = corr_matrix.shape
# ... iterate through all columns
for col in corr_matrix.columns:
    # ... iterate through all rows
    for row in corr_matrix.index:
        # ... if row and col are identical, set corr value to 0
        if row == col:
            corr_matrix.loc[row, col] = 0
            break

corr_matrix_sorted = corr_matrix.copy()

while len(corr_matrix_sorted.columns) > NR_TOP_FEATURES:
    # (Re) Calculate row correlation and sums
    highest_corrs = corr_matrix_sorted.max()
    sum_corrs = corr_matrix_sorted.sum()
    # tempolarily add two columns (highest, sum)
    corr_matrix_sorted['highest'] = highest_corrs
    corr_matrix_sorted['sum'] = sum_corrs
    # Sort corr_matrix rows by highest_corrs and overall sum_cors
    corr_matrix_sorted = corr_matrix_sorted.sort_values(
        by=['highest', 'sum'], ascending=True).drop(['highest', 'sum'], axis=1)

    # Get the name of the most correlated feature (last row)
    high_corr_feature = corr_matrix_sorted.index[-1]
    high_corr_value = corr_matrix_sorted[high_corr_feature].max()
    # print(f'Dropping: {high_corr_feature} - Correlation: {high_corr_value:.3f}')
    # ... drop the highest correlated feature from rows and columns
    corr_matrix_sorted.drop(high_corr_feature, axis=1, inplace=True)
    corr_matrix_sorted.drop(high_corr_feature, axis=0, inplace=True)

# Store names of remaining features
corr_selected_features = corr_matrix_sorted.columns

####
feature_cols = corr_selected_features
target_cols = df.drop((feature_cols), axis=1).columns
# Create X and y
X_train = df.loc[train_idx, feature_cols]
X_test = df.loc[test_idx, feature_cols]
y_train = df.loc[train_idx, target_cols]
y_test = df.loc[test_idx, target_cols]


# * Baseline Model (Random Forest)

# Create a multi-output RF baseline model
# ... (MultiOutputRegressor is not strictly necessary, but can increase performance)
rfr = MultiOutputRegressor(RandomForestRegressor(n_estimators=50, max_depth=8,
                                                 random_state=SEED, n_jobs=-1))

# Train the model (will take 5-15 seconds)
rfr.fit(X_train, y_train)

# Make predictions on the test set
y_pred = finalize_pred(pd.DataFrame(rfr.predict(X_test), columns=y_train.columns), 1, 5)

# Show some scores
print('\nBaseline Model (Multi Output - Random Forest)')
print(f'Model R2 Score: {r2_score(y_test, y_pred):.3f}')
print(f'Prediction Mean Squared Error: {mean_squared_error(y_test, y_pred):.3f}')
print(f'Prediction Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.3f}')


# * Baseline Model (Neural Network / MLP)

# Create a preprocessing step to scale all features to the range [0, 1]
scaler = MinMaxScaler()

# Create a multi-output MLP baseline model
mlp = MLPRegressor(hidden_layer_sizes=(256, 64, 16), alpha=0.01, activation='relu',
                   solver='adam', max_iter=1000, learning_rate='adaptive',
                   validation_fraction=0.15, n_iter_no_change=50, batch_size=64,
                   learning_rate_init=0.03, random_state=SEED)
# mlp = MultiOutputRegressor(MLPRegressor(hidden_layer_sizes=(16, 8, 4), alpha=0.01, activation='relu',
#                    solver='adam', max_iter=1000, learning_rate='adaptive',
#                    validation_fraction=0.15, n_iter_no_change=50, batch_size=32,
#                    learning_rate_init=0.03, random_state=SEED))

# Create a pipeline to chain the preprocessing step and the model
nn_pipe = make_pipeline(scaler, mlp)

# Train the model (will take 10-25 seconds)
nn_pipe.fit(X_train, y_train)

# Predict on the test set and show first 5 predictions
y_pred = finalize_pred(pd.DataFrame(
    nn_pipe.predict(X_test),
    columns=y_train.columns), 1, 5)

# Show some scores
print('\nBaseline Model (Neural Network - MLP)')
print(f'Model R2 Score: {r2_score(y_test, y_pred):.3f}')
print(f'Prediction Mean Squared Error: {mean_squared_error(y_test, y_pred):.3f}')
print(f'Prediction Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.3f}')


# %% Advanced model

# Define nr of targets

train_data = (df.loc[train_idx] - 1) / 4
test_data = (df.loc[test_idx] - 1) / 4

X_train = train_data[feature_cols]
y_train = train_data[target_cols]
X_test = test_data[feature_cols]
y_test = test_data[target_cols]
# define

# define the model architecture
model = Sequential()
model.add(Dense(256, input_dim=NR_TOP_FEATURES, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(y_test.shape[1], activation='linear'))

# compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Define early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=30)

# train the model
model.fit(
    X_train,
    y_train,
    epochs=1000,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop])


# Predict on the test set and show first 5 predictions
keras_results = pd.DataFrame(model.predict(X_test) * 4 + 1, columns=y_train.columns)
y_pred = finalize_pred(keras_results, 1, 5)
y_test = (y_test * 4) + 1
# Show some scores
print('\nBaseline Model (Neural Network - Keras)')
print(f'Model R2 Score: {r2_score(y_test, y_pred):.3f}')
print(f'Prediction Mean Squared Error: {mean_squared_error(y_test, y_pred):.3f}')
print(f'Prediction Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.3f}')


# %%


# Further approaches:
# https://scikit-learn.org/stable/auto_examples/cluster/plot_bisect_kmeans.html#sphx-glr-auto-examples-cluster-plot-bisect-kmeans-py
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.VarianceThreshold.html
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.corr.html
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html


# %% Corr plot

corr_matrix = df.corr().abs()

# Plot the correlation matrix
sns.heatmap(
    corr_matrix,
    cmap='Blues',
    annot=False,
    fmt='.2f',
    linewidths=0.5,
    annot_kws={
        "size": 8})



# %%

# PCA

# Initialize the PCA algorithm with 2 components
pca = PCA(n_components=2)

# Fit the PCA algorithm to the data
X_pca = pca.fit_transform(df_cluster)

# Calculate the mean of each data point
mean_values = np.mean(df_cluster, axis=1)


# Get the index names
index_names = df_cluster.index

# Plot the data with colors based on the mean value and index names for each point
fig, ax = plt.subplots(figsize=(24, 16), dpi=250)
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=mean_values)
ax.set_xlabel('PCA component 1')
ax.set_ylabel('PCA component 2')
ax.set_title('PCA plot with colors based on mean value')
cbar = plt.colorbar(scatter)
cbar.set_label('Mean value')
for i, name in enumerate(index_names):
    ax.text(X_pca[i, 0], X_pca[i, 1], str(i) + ': ' + name, fontsize=5)
plt.show()

# %%

# PCA

# Initialize the PCA algorithm with 2 components
pca = PCA(n_components=3)

# Fit the PCA algorithm to the data
X_pca = pca.fit_transform(df_cluster)

# Calculate the mean of each data point
mean_values = np.mean(df_cluster, axis=1)


# Get the index names
index_names = df_cluster.index

# Plot the data with colors based on the mean value and index names for each point
fig, ax = plt.subplots(figsize=(24, 16), dpi=250)
scatter = ax.scatter(X_pca[:, 2], X_pca[:, 1], c=mean_values)
ax.set_xlabel('PCA component 3')
ax.set_ylabel('PCA component 2')
ax.set_title('PCA plot with colors based on mean value')
cbar = plt.colorbar(scatter)
cbar.set_label('Mean value')
for i, name in enumerate(index_names):
    ax.text(X_pca[i, 2], X_pca[i, 1], str(i) + ': ' + name, fontsize=5)
plt.show()

# %%
