#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 19:12:40 2020

@author: Max
"""


import cpi
import pandas as pd 
import numpy as np
import quandl
from datetime import date
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss

#data
QUANDL_KEY = 'DxswD4b7N-71v3B7DaXk'

quandl.ApiConfig.api_key = QUANDL_KEY

df = quandl.get(dataset='WGC/GOLD_MONAVG_USD',
                start_date='2000-01-01',
                end_date='2011-12-31')

df.rename(columns={'Value': 'price'}, inplace=True)
#resample removes dublicates in the same month 
df = df.resample('M').last()

#Taken from chapter_3_utils.py of the github of python cookbook
def test_autocorrelation(x, n_lags=40, alpha=0.05, h0_type='c'):

    adf_results = adf_test(x)
    kpss_results = kpss_test(x, h0_type=h0_type)

    print('ADF test statistic: {:.2f} (p-val: {:.2f})'.format(adf_results['Test Statistic'],
                                                             adf_results['p-value']))
    print('KPSS test statistic: {:.2f} (p-val: {:.2f})'.format(kpss_results['Test Statistic'],
                                                              kpss_results['p-value']))

    fig, ax = plt.subplots(2, figsize=(16, 8))
    plot_acf(x, ax=ax[0], lags=n_lags, alpha=alpha)
    plot_pacf(x, ax=ax[1], lags=n_lags, alpha=alpha)

    return fig

def adf_test(x):
    indices = ['Test Statistic', 'p-value',
               '# of Lags Used', '# of Observations Used']
    adf_test = adfuller(x, autolag='AIC')
    results = pd.Series(adf_test[0:4], index=indices)
    for key, value in adf_test[4].items():
        results[f'Critical Value ({key})'] = value
    
    return results 

def kpss_test(x, h0_type='c'):
    indices = ['Test Statistic', 'p-value', '# of Lags']
    kpss_test = kpss(x, regression=h0_type)
    results = pd.Series(kpss_test[0:3], index=indices)
    for key, value in kpss_test[3].items():
        results[f'Critical Value ({key})'] = value
    
    return results 

#update cpi
#cpi.update()

#deflate gold prices
DEFL_DATE = date(2011, 12, 31)

df['dt_index'] = df.index.map(lambda x: x.to_pydatetime().date())
df['price_deflated'] = df.apply(lambda x: cpi.inflate(x.price, x.dt_index, DEFL_DATE),
                                axis=1)

df[['price', 'price_deflated']].plot(title='Gold Price (deflated)')
#using natural logarithm

WINDOW = 12
selected_columns = ['price_log', 'rolling_mean_log', 'rolling_std_log']

df['price_log'] = np.log(df.price_deflated)
df['rolling_mean_log'] = df.price_log.rolling(window=WINDOW).mean()
df['rolling_std_log'] = df.price_log.rolling(window=WINDOW).std()

df[selected_columns].plot(title='Gold Price (logged)')

#Apply differencing

selected_columns = ['price_log_diff', 'roll_mean_log_diff', 'roll_std_log_diff']
df['price_log_diff'] = df.price_log.diff(1)
df['roll_mean_log_diff'] = df.price_log_diff.rolling(WINDOW).mean()
df['roll_std_log_diff'] = df.price_log_diff.rolling(WINDOW).std()

df[selected_columns].plot(title='Gold Price (1ste differences)')

print(test_autocorrelation(df.price_log_diff.dropna()))