"""Module to download perform basic pre-processing on Dow Jones data."""

# Author(s): Thomas Glanzer
# Creation : May 2021
# License: MIT license

# %% Libs and Variables #########################

# Import required libs
import os
import pandas as pd

# Define historic data path
DATA_PATH = './data/downloads/'
EXPORT_PATH = './data/preprocessed/'
FILE_PREFIX = 'dow_historic'

# Define Nr of Digits for rounding
DIGITS = 2

# Define cutoff boundaries
LOWER_CUTOFF_YEAR = 2000
UPPER_CUTOFF_YEAR = 2020


# %% Code Execution #############################

# Get all files_list in dir
files_list = os.listdir(DATA_PATH)
# Only keep elements which are csv files_list
files_list = [element for element in files_list if element[-4:] == '.csv']

# Create empty df
df_all = pd.DataFrame()

# Iterate through all files_list in folder
print('# Data-Points #')
print()
for file in files_list:
    # ... read csv file
    df = pd.read_csv(DATA_PATH + file, sep=',', skipinitialspace=True)
    # ... strip whitespace in column names
    df = df.rename(columns=lambda x: x.strip())
    # ... rename cols
    df = df.rename(columns={'adjusted': 'adj_close'})
    # ... transform date to datetime
    df.date = pd.to_datetime(df.date, format="%Y-%m-%d")

    # ... add stock symbol name
    symbol = file.split('_')[0].split('.')[0]
    df['stock'] = symbol

    # ... round all floats
    df = round(df, DIGITS)

    # ... remove all dates which are out of range
    df = df.loc[df.date.dt.year <= UPPER_CUTOFF_YEAR]
    df = df.loc[df.date.dt.year >= LOWER_CUTOFF_YEAR]

    # ... filter for NAs and other conditions ...
    df = df.dropna(axis=0)
    # ... print data point stats in a godod-lookin fashion
    print(' ', str(symbol).ljust((5)), str(len(df)).rjust(5))

    # ... append current data to df
    df_all = df_all.append(df, ignore_index=True)


# Prepare export
df_all = round(df_all, DIGITS)
df_all['volume'] = df_all['volume'].astype('int64')

# Bring columns in the right oder
columns_ordered = ['stock', 'date', 'open', 'high', 'low', 'close',
                   'adj_close', 'volume', 'dividend', 'split']
df_all = df_all[columns_ordered]

# Sort values
df_all = df_all.sort_values(['stock', 'date'])
df_all = df_all.reset_index(drop=True)

# Export DF_all CSV
df_all.to_csv(EXPORT_PATH + FILE_PREFIX + '_%s_%s.csv'
              % (LOWER_CUTOFF_YEAR, UPPER_CUTOFF_YEAR), index=False)


# %%
