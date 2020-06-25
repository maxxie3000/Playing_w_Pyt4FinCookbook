#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 19:44:48 2020

@author: Max
"""

import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
import pmdarima as pm

stock = '^AEX'

df = yf.download(stock, 
                 start = '2000-01-01',
                 end = '2020-01-31',
                 adjusted = True,
                 progress = False)

goog = df.resample('W').last() \
    .rename(columns={'Adj Close':'adj_close'}).adj_close
    
arima = ARIMA(goog, 
              order=(2,1,1)).fit(disp=0)

auto_arima = pm.auto_arima(goog, 
                      error_action='ignore',
                      suppress_warnings=True,
                      seasonal=False,
                      stepwise=False,
                      approximation=False,
                      n_jobs = -1)

df = yf.download(stock,
                 start='2020-02-01',
                 end='2020-05-31',
                 adjusted=True,
                 progress=False)

test = df.resample('W').last() \
    .rename(columns={'Adj Close':'adj_close'}).adj_close
    
#Obtain forecasts
n_forecasts = len(test)

arima_pred = arima.forecast(n_forecasts)


arima_pred = [pd.DataFrame(arima_pred[0], columns=['prediction']),
              pd.DataFrame(arima_pred[2], columns=['ci_lower',
                                                  'ci_upper'])]

arima_pred = pd.concat(arima_pred, axis=1).set_index(test.index)

#Auto arima
auto_arima_pred = auto_arima.predict(n_periods=n_forecasts,
                                      return_conf_int=True,
                                      alpha=0.05)


auto_arima_pred = [pd.DataFrame(auto_arima_pred[0], columns=['prediction']),
              pd.DataFrame(auto_arima_pred[1], columns=['ci_lower',
                                                  'ci_upper'])]

auto_arima_pred = pd.concat(auto_arima_pred, axis=1).set_index(test.index)

#plotting
fig, ax = plt.subplots(1)

ax = sns.lineplot(data=test, label='Actual')
ax.plot(arima_pred.prediction, label='ARIMA(2,1,1)')
ax.fill_between(arima_pred.index,
                arima_pred.ci_lower,
                arima_pred.ci_upper,
                alpha=0.3)

ax.plot(auto_arima_pred.prediction, label='ARIMA(3,1,2)')
ax.fill_between(arima_pred.index,
                arima_pred.ci_lower,
                arima_pred.ci_upper,
                alpha=0.2)

ax.set(title="{0}'s stock price - actual vs. predicted".format(stock),
       xlabel='Date',
       ylabel='Price ($)')
ax.legend(loc='upper left')

plt.show()

