#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 12:33:59 2020

@author: Max
"""


import pandas as pd
import quandl
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

QUANDL_KEY = 'DxswD4b7N-71v3B7DaXk'

quandl.ApiConfig.api_key = QUANDL_KEY

df = quandl.get(dataset='WGC/GOLD_MONAVG_USD',
                start_date='2000-01-01',
                end_date='2011-12-31')

df.rename(columns={'Value': 'price'}, inplace=True)
#resample removes dublicates in the same month 
df = df.resample('M').last()

#Add rolling mean and SD
WINDOW_SIZE = 12
df['rolling_mean'] = df.price.rolling(window=WINDOW_SIZE).mean()
df['rolling_std'] = df.price.rolling(window=WINDOW_SIZE).std()
#Plot of price with rolling
#df.plot(title='Gold price')

#Seasonal decomposition
decomposition_results = seasonal_decompose(df.price, model='multiplicative')

decomposition_results.plot().suptitle('Multiplicative Decomposition', fontsize=18)

plt.show()