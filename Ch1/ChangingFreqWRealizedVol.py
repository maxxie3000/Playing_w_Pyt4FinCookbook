#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 16:00:55 2020

@author: Max
"""


import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt 

df = yf.download(['AAPL'],start='2000-01-01', end='2010-12-31',actions = 'inline' ,progress=True)
df = df.loc[:,['Adj Close']]
df.rename(columns={'Adj Close':'adj_close'}, inplace=True)
#Calculate simple and log return

df['log_rtn'] = np.log(df.adj_close/df.adj_close.shift(1))
df.pop('adj_close')
#Calculation for realized volatility
def realized_volatility(x):
    return np.sqrt(np.sum(x**2))
#Calculate monthly realized volatility
df_rv = df.groupby(pd.Grouper(freq='M')).apply(realized_volatility)
df_rv.rename(columns={'log_rtn':'rv'},inplace=True)

#Annualize 
df_rv.rv = df_rv.rv * np.sqrt(12)

#plot
fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(df)
ax[1].plot(df_rv)

