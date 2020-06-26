#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 20:31:28 2020

@author: Max
"""


import pandas as pd
import numpy as np
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
    #Only needs assets and weights for portfolio, others for traditional
    def __init__(self):
        print("Fama_French made")
        pass
    
    def set_values(self, risky_asset, start_date, end_date, assets=None , weights=None ):
        #individual asset
        self.risky_asset = risky_asset
        self.start_date = start_date
        self.end_date = end_date
        
        #portfolio 
        self.assets = assets
        self.weights = weights
        
    
    def load_factors_web(self):
        #for fun returns dictionary
        ff_dict = web.DataReader('F-F_Research_Data_Factors', 'famafrench', 
                                 start=self.start_date)
        self.factor_df = ff_dict[0].div(100)
        self.factor_df.index = self.factor_df.index.format()
        
    
    def load_factors_csv(self):
        #factors
        factor_df = pd.read_csv('F-F_Research_Data_Factors.CSV', skiprows = 3)
        str_to_match = ' Annual Factors: January-December '
        indices = factor_df.iloc[:,0] == str_to_match
        start_of_annual = factor_df[indices].index[0]
        
        self.factor_df = factor_df[factor_df.index < start_of_annual]
        
    def rename_columns(self):
        #rename columns
        self.factor_df.columns = ['date', 'mkt', 'smb', 'hml', 'rf']
        self.factor_df['date'] = pd.to_datetime(self.factor_df['date'],
                                           format='%Y%m').dt.strftime("%Y-%m")
        
        self.factor_df = self.factor_df.set_index('date')
        self.factor_df = self.factor_df.loc[self.start_date:self.end_date]
        self.factor_df = self.factor_df.apply(pd.to_numeric,
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
        
    def load_portfolio(self):
         self.portfolio_df = yf.download(self.assets,
                               start = self.start_date,
                               end = self.end_date,
                               adjusted = True,
                               progress = False)   
         self.portfolio_df = self.portfolio_df['Adj Close']\
             .resample('M').last()\
                 .pct_change().dropna()
         self.portfolio_df.index = self.portfolio_df.index.strftime('%Y-%m')
         self.portfolio_df['portfolio_returns'] = np.matmul(
             self.portfolio_df[self.assets].values, self.weights)
   
    def estimate_indv_model(self):
        self.load_factors_csv()
        self.rename_columns()
        self.load_asset()
        
        #merge
        ff_data = self.factor_df.join(self.y)
        ff_data['excess_rtn'] = ff_data.rtn - ff_data.rf
        
        #estimate
        ff_model = smf.ols(formula='excess_rtn ~ mkt + smb + hml', data = ff_data).fit()
        
        print(ff_model.summary())

        return ff_model
    
    def get_portfolio_data(self):
        self.load_portfolio()
        self.load_factors_web()
        
        self.ff_data = self.portfolio_df.join(self.factor_df).drop(self.assets,
                                                                   axis = 1)
        self.ff_data.columns = ['portf_rtn', 'mkt', 'smb', 'hml', 'rf']
        self.ff_data['portf_ex_rtn'] = self.ff_data.portf_rtn - self.ff_data.rf
        
    def rolling_factor_model(self, model_formula, window_size):
        coeffs = []
        self.get_portfolio_data()
        
        for start_index in range(len(self.ff_data) - window_size + 1):
            end_index = start_index + window_size
            
            ff_model = smf.ols(formula = model_formula, 
                               data=self.ff_data[start_index:end_index]).fit()
            coeffs.append(ff_model.params)
        
        self.coeffs_df = pd.DataFrame(coeffs, 
                                      index=self.ff_data.index[window_size - 1:])
        
        self.coeffs_df.plot(title= 'Rolling Fame-French Three-Factor model')
        plt.show()
        return self.coeffs_df
        
        
        
        

        
        
    

