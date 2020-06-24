#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 21:30:54 2020

@author: Max
"""

import matplotlib.pyplot as plt

import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
import scipy.stats as scs

import Autocorrelation

df = yf.download('GOOG', 
                 start = '2015-01-01',
                 end = '2018-12-31',
                 adjusted = True,
                 progress = False)

goog = df.resample('W').last() \
    .rename(columns={'Adj Close':'adj_close'}).adj_close
    
#applying first differences
goog_diff = goog.diff().dropna()

fig, ax = plt.subplots(2, sharex=True)
goog.plot(title= "Google's stock price", ax = ax[0])
goog_diff.plot(ax=ax[1], title = "First Differences")

print(Autocorrelation.test_autocorrelation(goog_diff))

arima = ARIMA(goog, order=(2,1,1)).fit(disp=0)
arima.summary()