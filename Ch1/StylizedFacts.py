#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 16:05:20 2020

@author: Max
"""
import numpy as np 
import pandas as pd
import yfinance as yf
import seaborn as sns
import scipy.stats as scs
import statsmodels.api as sm
import statsmodels.tsa.api as smt
import matplotlib.pyplot as plt

df = yf.download(['^GSPC'],start='1985-01-02',actions = 'inline' ,progress=True)
df = df.loc[:,['Adj Close']]
df.rename(columns={'Adj Close':'adj_close'}, inplace=True)
#Calculate simple and log return
df['simple_rtn'] = df.adj_close.pct_change()
df['log_rtn'] = np.log(df.adj_close/df.adj_close.shift(1))
df = df.iloc[1:]


#non-guassian distribution of returns
r_range = np.linspace(min(df.log_rtn), max(df.log_rtn), num=1000)
mu = df.log_rtn.mean()
sigma = df.log_rtn.std()
norm_pdf = scs.norm.pdf(r_range, loc=mu, scale=sigma)

#plotting
fig, ax = plt.subplots(1,2,figsize=(16,8))
#histogram
sns.distplot(df.log_rtn, kde=False, norm_hist=True, ax=ax[0])                                    
ax[0].set_title('Distribution of MSFT returns', fontsize=16)                                                    
ax[0].plot(r_range, norm_pdf, 'g', lw=2, 
           label=f'N({mu:.5f}, {sigma**2:.4f})')
ax[0].legend(loc='upper left');

# Q-Q plot
qq = sm.qqplot(df.log_rtn.values, line='s', ax=ax[1])
ax[1].set_title('Q-Q plot', fontsize = 16)

plt.show()

#Descriptive statistics
jb_test = scs.jarque_bera(df.log_rtn.values)

print('---------- Descriptive Statistics ----------')
print('Range of dates:', min(df.index.date), '-', max(df.index.date))
print('Number of observations:', df.shape[0])
print(f'Mean: {df.log_rtn.mean():.4f}')
print(f'Median: {df.log_rtn.median():.4f}')
print(f'Min: {df.log_rtn.min():.4f}')
print(f'Max: {df.log_rtn.max():.4f}')
print(f'Standard Deviation: {df.log_rtn.std():.4f}')
print(f'Skewness: {df.log_rtn.skew():.4f}')
print(f'Kurtosis: {df.log_rtn.kurtosis():.4f}') 
print(f'Jarque-Bera statistic: {jb_test[0]:.2f} with p-value: {jb_test[1]:.2f}')

#Visualize volatility clustering
df.log_rtn.plot(title='Daily MSFT returns', figsize=(10,6))

#Autocorrelation
#Parameters for AC plots
N_LAGS = 50
SIGNIFICANCE_LEVEL = 0.5

acf = smt.graphics.plot_acf(df.log_rtn,
                            lags = N_LAGS,
                            alpha = SIGNIFICANCE_LEVEL)
plt.tight_layout()
plt.show()

#Small and decreasing autocorrelation in squared/absolute returns 

#create ACF plots
gif, ax = plt.subplots(2, 1, figsize=(12,10))

#Squared
smt.graphics.plot_acf(df.log_rtn**2,
                      lags = N_LAGS,
                      alpha = SIGNIFICANCE_LEVEL,
                      ax = ax[0])
ax[0].set(title='Autocorrelation Plots',
          ylabel="Squared Returns")
#Absolute
smt.graphics.plot_acf(np.abs(df.log_rtn),
                      lags=N_LAGS,
                      alpha=SIGNIFICANCE_LEVEL,
                      ax = ax[1])
ax[1].set(ylabel = 'Absolute Returns',
          xlabel = 'lag')

plt.show()

#Leverage effect
#Calculate volatility measures
df['moving_std_252'] = df[['log_rtn']].rolling(window=252).std()
df['moving_std_21'] = df[['log_rtn']].rolling(window=21).std()

#plot
fig, ax = plt.subplots(3,1,figsize=(18,15),sharex=True)

df.adj_close.plot(ax=ax[0])
ax[0].set(title='MSFT time series',
          ylabel='Stock price ($)')

df.log_rtn.plot(ax=ax[1])
ax[1].set(ylabel='Log returns (%)')

df.moving_std_252.plot(ax=ax[2], color='r', 
                       label='Moving Volatility 252d')
df.moving_std_21.plot(ax=ax[2], color='g', 
                      label='Moving Volatility 21d')
ax[2].set(ylabel='Moving Volatility',
          xlabel='Date')
ax[2].legend()

# plt.tight_layout()
# plt.savefig('images/ch1_im15.png')
plt.show()
