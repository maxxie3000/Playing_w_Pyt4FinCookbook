#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 16:15:40 2020

@author: Max
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

class Monte_Carlo():
    def __init__(self):
        self.risky_asset = 'MSFT'
        self.start_date = '2018-01-01'
        self.end_date = '2019-07-31'
        
    def data(self):
        self.df = yf.download(self.risky_asset,
                         start=self.start_date,
                         end=self.end_date,
                         adjusted=True)
        #daily returns
        self.adj_close = self.df['Adj Close']
        self.returns = self.adj_close.pct_change().dropna()
        print(f'Average return: {100 * self.returns.mean():.2f}%')
        self.returns.plot(title=f'{self.risky_asset} returns: {self.start_date} - {self.end_date}')
        
        #train & test
        self.train = self.returns['2018-01-01':'2019-06-30']
        self.test = self.returns['2019-07-01':'2019-07-31']

    def MC(self):
        self.data()
        #specify parameters
        self.T = len(self.test)
        self.N = len(self.test)
        self.S_O = self.adj_close[self.train.index[-1].date()]
        self.n_sim = 100
        self.mu = self.train.mean()
        self.sigma = self.train.std()
        
        self.gbm_simulations = self.simulate_gbm()
    
        #preparing
        self.last_train_date = self.train.index[-1].date()
        self.first_test_date = self.test.index[0].date()
        self.last_test_date = self.test.index[-1].date()
        self.plot_title = (f'{self.risky_asset} Simulation '
                           f'({self.first_test_date}:{self.last_test_date})')

        self.selected_indices = self.adj_close[self.last_train_date:self.last_test_date].index
        self.index = [date.date() for date in self.selected_indices]

        self.gbm_simulations_df = pd.DataFrame(np.transpose(self.gbm_simulations), index = self.index)        
        
        #plotting
        self.ax = self.gbm_simulations_df.plot(alpha=0.2, legend=False)
        
        self.line_1, = self.ax.plot(self.index, self.gbm_simulations_df.mean(axis = 1),
                                    color='red')
        self.line_2, = self.ax.plot(self.index, self.adj_close[self.last_train_date:self.last_test_date],
                                    color='blue')
        self.ax.set_title(self.plot_title, fontsize=16)
        
        self.ax.legend((self.line_1, self.line_2), ('mean', 'actual'))
        plt.show()
        
    
    def simulate_gbm(self):
        self.dt = self.T / self.N
        self.dW = np.random.normal(scale = np.sqrt(self.dt),
                                   size = (self.n_sim, self.N))
        self.W = np.cumsum(self.dW, axis = 1)
       
        self.time_step = np.linspace(self.dt, self.T, self.N)

        self.time_steps = np.broadcast_to(self.time_step, (self.n_sim, self.N))

        self.S_t = self.S_O * np.exp((self.mu - 0.5 * self.sigma ** 2) * self.time_steps
                                     + self.sigma * self.W)
        self.S_t = np.insert(self.S_t, 0 , self.S_O, axis = 1)
        print(self.S_t)
        return self.S_t
    
        
        
        
        
x = Monte_Carlo()
x.MC()