#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 15:53:59 2020

@author: Max
"""

import numpy as np 
import pandas as pd
import yfinance as yf
import quandl  

#prices to returns
#Data download
df = yf.download(['AAPL'],start='2000-01-01', end='2010-12-31',actions = 'inline' ,progress=True)
df = df.loc[:,['Adj Close']]
df.rename(columns={'Adj Close':'adj_close'}, inplace=True)
#Calculate simple and log return
df['simple_rtn'] = df.adj_close.pct_change()
df['log_rtn'] = np.log(df.adj_close/df.adj_close.shift(1))
#real return adjusted by inflation
df_all_dates = pd.DataFrame(index=pd.date_range(start='1999-12-31',
                                                end = '2010-12-31'))
df = df_all_dates.join(df[['adj_close']], how='left') \
    .fillna(method='ffill') \
    .asfreq('M')

QUANDL_KEY = 'DxswD4b7N-71v3B7DaXk'
quandl.ApiConfig.api_key = QUANDL_KEY

#Consumer price index
df_cpi = quandl.get(dataset='RATEINF/CPI_USA',
                    start_date = '1999-12-01',
                    end_date ='2010-12-31')
df_cpi.rename(columns={'Value':'cpi'}, inplace=True)
#merge dfs
df_merged = df.join(df_cpi, how='left')
#calculate returns and inflation rates
df_merged['simple_rtn'] = df_merged.adj_close.pct_change()
df_merged['inflation_rate'] = df_merged.cpi.pct_change()
#Adjust returns for inflation
df_merged['real_rtn'] = (df_merged.simple_rtn + 1) / (df_merged.inflation_rate + 1) - 1
