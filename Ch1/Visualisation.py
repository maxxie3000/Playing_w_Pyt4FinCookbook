#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 16:18:41 2020

@author: Max
"""

import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt 
import numpy as np
import cufflinks as cf
from plotly.offline import iplot, init_notebook_mode

#Data set
df = yf.download(['MSFT'],
                 
                 actions = 'inline',
                 progress=True
                 )
df = df.loc[:,['Adj Close']]
df.rename(columns={'Adj Close':'adj_close'}, inplace=True)
#Returns
df['simple_rtn'] = df.adj_close.pct_change()
df['log_rtn'] = np.log(df.adj_close/df.adj_close.shift(1))

#Plot
fig, ax = plt.subplots(3,1, figsize=(24,20), sharex=True)

df.adj_close.plot(ax=ax[0])
ax[0].set(title = 'MSFT time series',
          ylabel = 'Stock price ($)')

df.simple_rtn.plot(ax=ax[1])
ax[1].set(ylabel = 'Simple returns (%)')

df.log_rtn.plot(ax=ax[2])
ax[2].set(xlabel = 'Date',
          ylabel = 'Log returns (%)')

#Plotting using Plotly and Cufflinks, NOTE:only in Jupyter Notebook!!!
#run once
#cf.set_config_file(world_readable = True, theme = 'pearl', offline = True)
init_notebook_mode()

df.iplot(subplots=True, shape=(3,1), shared_xaxes=True, title = 'MSFT time series')