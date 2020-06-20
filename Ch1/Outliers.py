#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 16:38:30 2020

@author: Max
"""

import numpy as np 
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt 


df = yf.download(['AAPL'],start='2000-01-01', end='2010-12-31',actions = 'inline' ,progress=True)
df = df.loc[:,['Adj Close']]
df.rename(columns={'Adj Close':'adj_close'}, inplace=True)
#Calculate simple and log return
df['simple_rtn'] = df.adj_close.pct_change()
df['log_rtn'] = np.log(df.adj_close/df.adj_close.shift(1))

#Calculate rolling mean and std
df_rolling = df[['simple_rtn']].rolling(window=21).agg(['mean','std'])
df_rolling.columns = df_rolling.columns.droplevel()
#Join rolling data with orginal data
df_outliers = df.join(df_rolling)
#function for outliers 
def identify_outliers(row,n_sigmas=3):
    x = row['simple_rtn']
    mu = row['mean']
    sigma = row['std']
    if (x > mu + 3 * sigma) | (x < mu - 3 * sigma):
        return 1 
    else:
        return 0 

#identify outliers and extract values
df_outliers['outlier'] = df_outliers.apply(identify_outliers,axis=1)
outliers = df_outliers.loc[df_outliers['outlier']==1,['simple_rtn']]

#plot results
fig, ax = plt.subplots() 
ax.plot(df_outliers.index, df_outliers.simple_rtn, color = 'blue', label = 'Normal')
ax.scatter(outliers.index, outliers.simple_rtn, color = "red", label='Anomaly')
ax.set_title("Apple's stock returns")   
ax.legend(loc='lower right')
    
    