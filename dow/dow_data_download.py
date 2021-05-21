"""Module to download Dow Jones data."""

# Author(s): Thomas Glanzer
# Creation : May 2021
# License: MIT license

# References
# https://github.com/RomelTorres/alpha_vantage

# %% Libs and Variables #########################

# Import required libs
import time
from alpha_vantage.timeseries import TimeSeries

# Define output path
DATA_PATH = './data/downloads/'

# Obtain API key from disk
API_KEY_LOC = '../../alpha_vantage.key'
API_KEY = open(API_KEY_LOC, mode='r').read()
API_SLEEP_TIMER = 12

# Define Time Series with API Key, Output format and indexing type
ts = TimeSeries(key=API_KEY, output_format='pandas', indexing_type='integer')

# Define stocks list(s) (Dow Jones 30)
stocks_list = sorted(['HD', 'MSFT', 'GS', 'JPM', 'V', 'CVX', 'MRK', 'AAPL', 'VZ', 'AMGN', 'CRM',
                      'PG', 'DIS', 'TRV', 'IBM', 'CAT', 'MCD', 'INTC', 'MMM', 'WBA', 'AXP', 'HON',
                      'WMT', 'BA', 'KO', 'UNH', 'JNJ', 'DOW', 'CSCO', 'NKE'])


# %% Code Execution #############################

# Make API request for each element in list
print('Starting download for', len(stocks_list), 'stocks ...')
print()
# ... iterate through stocks
for idx, stock in enumerate(stocks_list):

    # Mark tech companies
    print('Downloading', stock, '...')

    # Define API Request
    data, _ = ts.get_daily_adjusted(stock, 'full')    # pylint: disable=unbalanced-tuple-unpacking

    # Rename base columns
    base_cols = ['date', 'open', 'high', 'low', 'close', 'adjusted', 'volume', 'dividend', 'split']
    data.columns = base_cols

    # Save Data in Output Folder
    data.to_csv(DATA_PATH + stock + '_base.csv', index=False)          # pylint: disable=no-member

    # Sleep if to avoid api-overload
    time.sleep(API_SLEEP_TIMER)

print()
print('Downloads finished!')


# %%
