#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 18:42:33 2020

@author: Max
"""
import quandl
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss
import pandas as pd

#create dataframe
QUANDL_KEY = 'DxswD4b7N-71v3B7DaXk'

quandl.ApiConfig.api_key = QUANDL_KEY

df = quandl.get(dataset='WGC/GOLD_MONAVG_USD',
                start_date='2000-01-01',
                end_date='2011-12-31')

df.rename(columns={'Value': 'price'}, inplace=True)
#resample removes dublicates in the same month 
df = df.resample('M').last()

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

#generate ACF/PACF plots
n_lags = 40 
significance_level = 0.05

fig,ax = plt.subplots(2,1)
plot_acf(df.price, ax=ax[0], lags=n_lags, alpha=significance_level)
plot_pacf(df.price, ax=ax[1], lags=n_lags, alpha=significance_level)

plt.show()

