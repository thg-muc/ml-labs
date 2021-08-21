"""Basic Explorative Data Analysis von Dow Jones Stock Data."""

# Author(s): Thomas Glanzer
# Creation : May 2021
# License: MIT license


# %% Libs and Variables #########################

# Import required libs
import pandas as pd

# Define basic vars
DATA_PATH = './data/preprocessed/'
FILE_NAME = 'dow_historic_2000_2020.csv'


# Read stock data
df = pd.read_csv(DATA_PATH + FILE_NAME)

# %% ####### Start with EDA


# What are the column dtypes and number of NAs ?
print(df.info())

# How many different stocks are in this data set ?
print(df.stock.nunique())

# What are the stock names and the number of trading days for each stock ?
print(df.stock.value_counts())

# What is the earliest trading date in the data ?
print(df.date.min())

# What is the latest trading date in the data ?
print(df.date.max())


# %%
