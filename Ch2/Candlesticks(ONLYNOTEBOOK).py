#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 19:14:00 2020

@author: Max
"""


import pandas as pd 
import yfinance as yf

df_twtr = yf.download('TWTR',
                      start='2018-01-01',
                      end='2018-12-31',
                      progress=False,
                      auto_adjust=True)

#building candlesticks 
import cufflinks as cf
from plotly.offline import iplot, init_notebook_mode 

init_notebook_mode()

qf = cf.QuantFig(df_twtr, title="Twitter's stock price",
                 legend='top', name='TWTR')

qf.add_volume()
qf.add_sma(periods=20, column='Close', color = 'red')
qf.add_ema(periods=20,color='green')

qf.iplot()