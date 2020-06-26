#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 19:17:53 2020

@author: Max
"""


import pandas as pd
import yfinance as yf
from arch import arch_model
import matplotlib.pyplot as plt
import sys 
sys.path.insert(0, '/Users/Max/Documents/GitHub/Playing_w_Pyt4FinCookbook/Ch4')

from Chapter4Classes import Fama_French
#ARCH models are for residuals! Not the returns 
class ARCH:
    def __init__(self):
        self.risky_asset = 'BTC-USD'
        self.start_date = '2015-01-01'
        self.end_date = '2018-12-31'
        #uses Fama_French model
        self.model_FF = Fama_French()
        self.model_FF.set_values(risky_asset = self.risky_asset, start_date = self.start_date, end_date= self.end_date)
        
    def data(self):
        self.df = yf.download(self.risky_asset,
                              start = self.start_date,
                              end= self.end_date,
                              adjusted = True)
        
        self.returns = 100* self.df['Adj Close'].pct_change().dropna()    
        self.returns.name = 'asset_returns'    
        
    def model_residuals_FF(self):
        self.results_FF = self.model_FF.estimate_indv_model()
        self.residuals = self.results_FF.resid * 100
      
        self.model_A = arch_model(self.residuals, mean='Zero', vol='ARCH', 
                                p = 1, o = 0, q = 0)
        self.model_fitted = self.model_A.fit(disp='off')
        print(self.model_fitted.summary())
        self.model_fitted.plot(annualize='D')
        plt.show()
    
    def model(self):
        self.data()
      
        self.model_A = arch_model(self.returns, mean='Zero', vol='ARCH', 
                                p = 1, o = 0, q = 0)
        
        self.model_fitted = self.model_A.fit(disp='off')
        print(self.model_fitted.summary())
        self.model_fitted.plot(annualize='D')
        plt.show()
        

x = ARCH()
x.model()
    