#!/usr/bin/env python

'''
    Task: Show performance variations between 2x US listed equities
    
    FINS3648 market prices using Yahoo
        1. Load market prices
        2. Analyze timeseries
        3. Show visuals
        4. Compare performance
        5. ...so what? 
    License: ""
'''

# Import initial libraries
from pandas_datareader import data as pdr
import fix_yahoo_finance as yf
import datetime
import matplotlib.pyplot as plt

# Set date ranges
start_sp = datetime.datetime(2017, 1, 1)
end_sp = datetime.datetime(2018, 8, 1)

# pandas reader as pdr and use uf as yahoo call
yf.pdr_override('tickers')
stock1 = pdr.get_data_yahoo('AAPL', start_sp, end_sp)
stock2 = pdr.get_data_yahoo('V', start_sp, end_sp)

print(stock1)
print(stock2)

# Extract narket close prices from dataseries
stock1_close = stock1.Close
stock1_diff = stock1.Close.pct_change()
stock1_roll = stock1_close.rolling(window=30)
stock1_roll_mean = stock1_roll.mean()

stock2_close = stock2.Close
stock2_diff = stock2.Close.pct_change()
stock2_roll = stock2_close.rolling(window=30)
stock2_roll_mean = stock2_roll.mean()

# Plot in combined views
plt.figure(1)

plt.subplot(221)
stock1_close.plot(c="b", markersize=8, label="Stock1 ",title='Stock1')
stock1_roll_mean.plot(color='red')

plt.subplot(222)
stock1_diff.plot(c="r", markersize=8, label="Stock1", title='Stock1 %Diff')

plt.subplot(223)
stock2_close.plot(c="b", markersize=8, label="Stock2 ",title='Stock2')
stock2_roll_mean.plot(color='red')

plt.subplot(224)
stock2_diff.plot(c="r", markersize=8, label="Stock2 Diff", title='Stock2 %Diff')

plt.show()