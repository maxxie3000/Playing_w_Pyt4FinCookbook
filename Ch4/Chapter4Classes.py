#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 20:31:28 2020

@author: Max
"""


import pandas as pd
import yfinance as yf
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from pandas_datareader.famafrench import get_available_datasets
import pandas_datareader.data as web

class CAPM:
    def __init__(self):
        self.risky_asset = 'AMZN'
        self.market_benchmark = '^GSPC'
        self.start_date = '2014-01-01'
        self.end_date = '2018-12-31'

    def calculate(self):
        self.get_data()
        #Beta calculation
        covariance = self.X.cov().iloc[0,1]
        benchmark_variance = self.X.market.var()
        beta = covariance / benchmark_variance
        
        Y = self.X.pop('asset')
        X = sm.add_constant(self.X)
        
        capm_model = sm.OLS(Y,X).fit()
        print(capm_model.summary())

    def get_data(self):
        df = yf.download([self.risky_asset, self.market_benchmark], 
                         start=self.start_date,
                         end=self.end_date,
                         adjusted=True,
                         progress=False)
        
        self.X = df['Adj Close'].rename(columns={self.risky_asset: 'asset', 
                                            self.market_benchmark: 'market'})\
            .resample('M').last().pct_change().dropna()
            
    def get_rf(self):
        n_days = 90
        
        df_rf = yf.download('^IRX',
                            start=self.start_date,
                            end=self.end_date)
        #resample to monthly frequency
        rf = df_rf.resample('M').last().Close / 100
        
        rf = ( 1 / (1 - rf * n_days / 360) ) ** (1/ n_days)
        rf = (rf ** 30) - 1
        
        rf.plot(title='Risk-free rate (13 week Treasury Bill)')
        plt.show()
        
    def get_rf_secondary(self):
        rf = web.DataReader('TB3MS', 
                            'fred', 
                            start=self.start_date,
                            end=self.end_date)
        
        rf = ( 1 + ( rf / 100)) ** (1/12) - 1
        rf.plot(title = 'Risk-free rate (3-Month Treasury Bill)')
        plt.show()        
        
class Fama_French:
    def __init__(self):
        self.risky_asset = 'FB'
        self.start_date = '2013-12-31'
        self.end_date = '2018-12-31'
    
    def load_factors_web(self):
        #for fun returns dictionary
        ff_dict = web.DataReader('F-F_Research_Data_Factors', 'famafrench', 
                                 start=self.start_date)
        self.factor_df = ff_dict[0]
        
    
    def load_factors_csv(self):
        #factors
        self.factor_df = pd.read_csv('F-F_Research_Data_Factors.CSV', skiprows = 3)
        str_to_match = ' Annual Factors: January-December '
        indices = factor_df.iloc[:,0] == str_to_match
        start_of_annual = factor_df[indices].index[0]
        
        self.factor_df = factor_df[factor_df.index < start_of_annual]
        
    def rename_columns(self):
        #rename columns
        self.factor_df.columns = ['date', 'mkt', 'smb', 'hml', 'rf']
        self.factor_df['date'] = pd.to_datetime(factor_df['date'],
                                           format='%Y%m').dt.strftime("%Y-%m")
        
        self.factor_df = factor_df.set_index('date')
        self.factor_df = factor_df.loc[self.start_date:self.end_date]
        self.factor_df = factor_df.apply(pd.to_numeric,
                                         errors='coerce').div(100)
    
    def load_asset(self):
        asset_df = yf.download(self.risky_asset,
                               start = self.start_date,
                               end = self.end_date,
                               adjusted = True)
        
        y = asset_df['Adj Close'].resample('M').last().pct_change().dropna()
        
        y.index = y.index.strftime('%Y-%m')
        y.name = 'rtn'
        self.y = y
        
   
    def estimate_model(self):
        self.load_factors_web()
        self.rename_columns()
        self.load_asset()
        
        #merge
        ff_data = self.factor_df.join(self.y)
        ff_data['excess_rtn'] = ff_data.rtn - ff_data.rf
        
        #estimate
        ff_model = smf.ols(formula='excess_rtn ~ mkt + smb + hml', data = ff_data).fit()
        
        print(ff_model.summary())
        
        
        
        
        
        
x = Fama_French()
x.estimate_model()
        
        
    

