#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 20:08:04 2020

@author: Max
"""


import matplotlib.pyplot as plt
import seaborn as sns 

#plt.set_cmap('cubehelix')
#sns.set_palette('cubehelix')

COLORS = ['b', 'g', 'r', 'c'] #[plt.cm.cubehelix(x) for x in [0.1, 0.3, 0.5, 0.7]]

#libraries
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date
from statsmodels.tsa.holtwinters import (ExponentialSmoothing,SimpleExpSmoothing,Holt)

df = yf.download('GOOG',
                 start = '2010-01-01',
                 end = '2018-12-31',
                 adjusted = True,
                 progress = False)

#aggreagte to monthly frequency 

goog = df.resample('M').last() \
    .rename(columns={'Adj Close': 'adj_close'}).adj_close
#create split
train_indices = goog.index.year < 2018
goog_train = goog[train_indices]
goog_test = goog[~train_indices]

test_length = len(goog_test)

#fit three SES models 
ses_1 = SimpleExpSmoothing(goog_train).fit(smoothing_level=0.2)
ses_forecast_1 = ses_1.forecast(test_length)

ses_2 = SimpleExpSmoothing(goog_train).fit(smoothing_level=0.5)
ses_forecast_2 = ses_2.forecast(test_length)

ses_3 = SimpleExpSmoothing(goog_train).fit()
alpha = ses_3.model.params['smoothing_level']
ses_forecast_3 = ses_3.forecast(test_length)

#plotting
goog.plot(color=COLORS[0],
          title = 'Simple Exponential Smoothin',
          label = 'Actual',
          legend = True )

ses_forecast_1.plot(c=COLORS[1], legend=True, 
                    label=r'$\alpha=0.2$')
ses_1.fittedvalues.plot(c=COLORS[1])

ses_forecast_2.plot(c=COLORS[2], legend = True, 
                    label=r'$\alpha=0.5$')
ses_2.fittedvalues.plot(c=COLORS[2])

ses_forecast_3.plot(c=COLORS[3], legend = True, 
                    label=r'$\alpha={0:.4f}$'.format(alpha))
ses_3.fittedvalues.plot(c=COLORS[3])

plt.show()
#For some reason from colors in legend

#Holt's variants:
# Holt's model with linear trend
hs_1 = Holt(goog_train).fit()
hs_forecast_1 = hs_1.forecast(test_length)

# Holt's model with exponential trend
hs_2 = Holt(goog_train, exponential=True).fit()
# equivalent to ExponentialSmoothing(goog_train, trend='mul').fit()
hs_forecast_2 = hs_2.forecast(test_length)

# Holt's model with exponential trend and damping
hs_3 = Holt(goog_train, exponential=False, 
            damped=True).fit(damping_slope=0.99)
hs_forecast_3 = hs_3.forecast(test_length)

goog.plot(color=COLORS[0],
          title="Holt's Smoothing models",
          label='Actual',
          legend=True)

hs_1.fittedvalues.plot(color=COLORS[1])
hs_forecast_1.plot(color=COLORS[1], legend=True, 
                   label='Linear trend')

hs_2.fittedvalues.plot(color=COLORS[2])
hs_forecast_2.plot(color=COLORS[2], legend=True, 
                   label='Exponential trend')

hs_3.fittedvalues.plot(color=COLORS[3])
hs_forecast_3.plot(color=COLORS[3], legend=True, 
                   label='Exponential trend (damped)')



