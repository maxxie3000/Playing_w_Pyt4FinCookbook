#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 19:31:50 2020

@author: Max
"""


import pmdarima as pm
import yfinance as yf
import pandas as pd 


df = yf.download('GOOG', 
                 start = '2015-01-01',
                 end = '2018-12-31',
                 adjusted = True,
                 progress = False)

goog = df.resample('W').last() \
    .rename(columns={'Adj Close':'adj_close'}).adj_close

model = pm.auto_arima(goog, 
                      error_action='ignore',
                      suppress_warnings=True,
                      seasonal=False,
                      stepwise=False,
                      approximation=False,
                      n_jobs = -1)

print(model.summary())

