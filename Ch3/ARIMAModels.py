#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 21:30:54 2020

@author: Max
"""

import matplotlib.pyplot as plt
import seaborn as sns 
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

#print(Autocorrelation.test_autocorrelation(goog_diff))

arima = ARIMA(goog, order=(2,1,1)).fit(disp=0)
#print(arima.summary())

def arima_diagnostics(resids, n_lags=40):
    #create placeholder subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
    
    r=resids
    resids = ( r - np.nanmean(r)) / np.nanstd(r)
    resids_nonmissing = resids[~(np.isnan(resids))]
    
    #residuals overt time
    sns.lineplot(x=np.arange(len(resids)), y=resids, ax=ax1)
    ax1.set_title('Standardized residuals')
    
    #distribution
    x_lim = (-1.96 * 2, 1.96 * 2)
    r_range = np.linspace(x_lim[0], x_lim[1])
    norm_pdf = scs.norm.pdf(r_range)
    sns.distplot(resids_nonmissing, hist=True, kde=True,
                 norm_hist=True, ax=ax2)
    ax2.plot(r_range, norm_pdf, 'g', lw=2, label='N(0,1)')
    ax2.set_title('Distribution of standardized residuals')
    ax2.set_xlim(x_lim)
    ax2.legend()
    
    #Q-Qplot
    qq = sm.qqplot(resids_nonmissing, line='s', ax=ax3)
    ax3.set_title('Q-Q plot')
    
    #ACF Plot
    plot_acf(resids, ax=ax4, lags=n_lags, alpha=0.05)
    ax4.set_title('ACF plot')
    
    return fig 

arima_diagnostics(arima.resid, 40)

#plt.show()

ljung_box_results = acorr_ljungbox(arima.resid)

fig, ax = plt.subplots(1, figsize=[16, 5])
sns.scatterplot(x=range(len(ljung_box_results[1])),
                y=ljung_box_results[1],
                ax=ax)
ax.axhline(0.05, ls='--', c='r')
ax.set(title='ljung-box tests results',
       xlabel = 'lag',
       ylabel = 'p-value')

plt.show()