#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 13:00:17 2020

@author: Max
"""


import pandas as pd
import seaborn as sns
import quandl
from fbprophet import Prophet
import matplotlib.pyplot as plt

QUANDL_KEY = 'DxswD4b7N-71v3B7DaXk'

quandl.ApiConfig.api_key = QUANDL_KEY

df = quandl.get(dataset='WGC/GOLD_DAILY_USD',
                start_date='2000-01-01',
                end_date='2005-12-31')

df.reset_index(drop=False, inplace=True)
df.rename(columns={'Date': 'ds', 'Value': 'y'}, inplace = True)

#split
train_indices = df.ds.apply(lambda x: x.year) < 2005

df_train = df.loc[train_indices].dropna()
df_test = df.loc[~train_indices].reset_index(drop=True)

#create model
model_prophet = Prophet(seasonality_mode='additive')
model_prophet.add_seasonality(name='monthly', period=30.5, fourier_order = 5)
#fit to train < 2005
model_prophet.fit(df_train)

#Forecasts
df_future = model_prophet.make_future_dataframe(periods=365)
df_pred = model_prophet.predict(df_future)
#Plotting
#model_prophet.plot(df_pred)

#model_prophet.plot_components(df_pred)
#plt.show()


#Basic performance evaluation
selected_columns = ['ds', 'yhat_lower', 'yhat_upper', 'yhat']

df_pred = df_pred.loc[:,selected_columns].reset_index(drop=True)
df_test = df_test.merge(df_pred, on=['ds'], how='left')
df_test.ds = pd.to_datetime(df_test.ds)
df_test.set_index('ds', inplace=True)

fig, ax = plt.subplots(1,1)

ax = sns.lineplot(data=df_test[['y', 'yhat_lower', 'yhat_upper', 'yhat']])

ax.fill_between(df_test.index,
                df_test.yhat_lower,
                df_test.yhat_upper,
                alpha=0.3)
ax.set(title='Gold Price - actual vs. predicted',
       xlabel = 'Date',
       ylabel = 'Gold price ($)')

plt.show()