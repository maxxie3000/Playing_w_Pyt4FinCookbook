#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 19:17:53 2020

@author: Max
"""


import pandas as pd
import numpy as np
import yfinance as yf
from arch import arch_model
import matplotlib.pyplot as plt
import sys 
sys.path.insert(0, '/Users/Max/Documents/GitHub/Playing_w_Pyt4FinCookbook/Ch4')

from Chapter4Classes import Fama_French
#ARCH models are for residuals! Not the returns 
class ARCH:
    def __init__(self):
        self.risky_asset = ['GOOG', 'MSFT', 'AAPL']
        self.start_date = '2015-01-01'
        self.end_date = '2018-12-31'
        self.N = len(self.risky_asset)
        
    def data(self):
        self.df = yf.download(self.risky_asset,
                              start = self.start_date,
                              end= self.end_date,
                              adjusted = True)
        
        self.returns = 100* self.df['Adj Close'].pct_change().dropna()    
        self.returns.plot(subplots=True, title=f'Stock returns: {self.start_date} - {self.end_date}')
    
    def model(self, risky_asset=None):
        if not risky_asset == None:
            self.risky_asset = risky_asset
            
        self.data()
      
        self.model_A = arch_model(self.returns, mean='Zero', vol='GARCH', 
                                p = 1, o = 0, q = 1)
        
        self.model_fitted = self.model_A.fit(disp='off')
        print(self.model_fitted.summary())
        self.model_fitted.plot(annualize='D')
        plt.show()
        
    def CCC_GARCH(self):
        self.data()
        
        #set storing lists
        self.coeffs = []
        self.cond_vol = []
        self.std_resids = []
        self.models = []
        
        #estimate univeriate Garch Models 
        for asset in self.returns.columns:
            model = arch_model(self.returns[asset], mean = 'constant',
                               vol='GARCH', p = 1, o = 0, q = 1).fit(update_freq=0,
                                                                     disp = 'off')
            self.coeffs.append(model.params)
            self.cond_vol.append(model.conditional_volatility)
            self.std_resids.append(model.resid / model.conditional_volatility)
            self.models.append(model)
            
        self.coeffs_df = pd.DataFrame(self.coeffs, index=self.returns.columns)
        self.cond_vol_df = pd.DataFrame(self.cond_vol).transpose().set_axis(self.returns.columns,
                                                                            axis = 'columns',
                                                                            inplace = False)
        self.std_resid_df = pd.DataFrame(self.std_resids).transpose().set_axis(self.returns.columns,
                                                                               axis='columns',
                                                                               inplace = False)
        print(self.coeffs_df)
        
        self.R = self.std_resid_df.transpose().dot(self.std_resid_df).div(len(self.std_resid_df))
       
        print(self.R)
        
        self.diag = []
        self.D = np.zeros((self.N, self.N))
        
        
        for model in self.models:
            self.diag.append(model.forecast(horizon=1).variance.values[-1][0])
        self.diag = np.sqrt(np.array(self.diag))
        np.fill_diagonal(self.D,  self.diag)
        
        print(self.D)
        print(self.R.values)
        
        self.H = np.matmul(np.matmul(self.D, self.R.values), self.D)
        
        print(self.H)
            
        
        

x = ARCH()
x.CCC_GARCH()
x.model('AAPL')
    