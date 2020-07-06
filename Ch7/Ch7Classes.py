#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 12:13:52 2020

@author: Max
"""

import yfinance as yf
import numpy as np
import pandas as pd
import pyfolio as pf 

import scipy.optimize as sco

import cvxpy as cp

import matplotlib.pyplot as plt
import seaborn as sns 

class equal_weight:
    def __init__(self):
        self.risky_assets = ['AAPL', 'IBM', 'MSFT', 'TWTR']
        self.start_date = '2017-01-01'
        self.end_date = '2018-12-31'
       

        
        self.n_assets = len(self.risky_assets)
    
    def data(self):
        self.prices_df = yf.download(self.risky_assets, 
                                     start = self.start_date,
                                     end = self.end_date,
                                     adjusted = True)
        
        self.returns = self.prices_df['Adj Close'].pct_change().dropna()
    
    def calculate(self):
        
        self.data()
        
        self.portfolio_weights = self.n_assets * [1 / self.n_assets]

        self.portfolio_returns = pd.Series(np.dot(self.portfolio_weights, self.returns.T),
                                           index=self.returns.index)
        
        pf.create_simple_tear_sheet(self.portfolio_returns)

class Efficient_Frontier:
    def __init__(self):
        self.n_portfolios = 10 ** 5
        self.n_days = 252
        self.risky_assets = ['FB', 'TSLA', 'TWTR', 'MSFT']
        self.risky_assets.sort()
        self.start_date = '2018-01-01'
        self.end_date = '2018-12-31'
        self.marks = ['o', 'X', 'd', '<']
        
        self.n_assets = len(self.risky_assets)
        
    def data(self):          
        self.prices_df = yf.download(self.risky_assets,
                                     start=self.start_date,
                                     end=self.end_date,
                                     adjusted = True)
        self.returns_df = self.prices_df['Adj Close'].pct_change().dropna()
        self.avg_returns = self.returns_df.mean() * self.n_days 
        self.cov_mat = self.returns_df.cov() * self.n_days
       
        
    def portfolio_metrics(self):
        #set weights
        np.random.seed(42)
        self.weights = np.random.random(size=(self.n_portfolios, self.n_assets))
        self.weights /= np.sum(self.weights, axis=1)[:, np.newaxis]
        #metrics
        self.portf_rtns = np.dot(self.weights, self.avg_returns)
        
        self.portf_vol = []
        for i in range(0, len(self.weights)):
            self.portf_vol.append(np.sqrt(np.dot(self.weights[i].T,
                                                 np.dot(self.cov_mat, self.weights[i]))))
        
        self.port_vol = np.array(self.portf_vol)
        self.portf_sharpe_ratio = self.portf_rtns / self.portf_vol

        self.portf_results_df = pd.DataFrame({'returns': self.portf_rtns,
                                              'volatility': self.portf_vol,
                                              'sharpe_ratio': self.portf_sharpe_ratio})  
    
    def efficient_frontier(self):

        self.n_points = 100
        self.portf_vol_ef = []
        self.indices_to_skip = []
        
        self.portf_rtns_ef = np.linspace(self.portf_results_df.returns.min(),
                                         self.portf_results_df.returns.max(),
                                         self.n_points)
        self.portf_rtns_ef = np.round(self.portf_rtns_ef, 2)
        
        self.portf_rtns = np.round(self.portf_rtns, 2)
        
        for point_index in range(self.n_points):
            if self.portf_rtns_ef[point_index] not in self.portf_rtns:
                self.indices_to_skip.append(point_index)
                continue
            self.matched_ind = np.where(self.portf_rtns == self.portf_rtns_ef[point_index])
            
            self.portf_vol_ef.append(np.min(self.port_vol[self.matched_ind]))
            
        self.portf_rtns_ef = np.delete(self.portf_rtns_ef, self.indices_to_skip)
    
    def plot(self):
        self.data()
        self.portfolio_metrics()
        self.efficient_frontier()
        self.sharpe_max()
        
        
        fig, ax = plt.subplots()
        self.portf_results_df.plot(kind='scatter', x='volatility',
                                   y='returns', c='sharpe_ratio',
                                   cmap='RdYlGn', edgecolors='black',
                                   ax=ax)
        ax.set(xlabel='Volatility',
               ylabel='Expected Returns',
               title='Efficient Frontier')
        ax.plot(self.portf_vol_ef, self.portf_rtns_ef, 'b--')
        for asset_index in range(self.n_assets):
            ax.scatter(x=np.sqrt(self.cov_mat.iloc[asset_index, asset_index]),
                       y=self.avg_returns[asset_index],
                       marker=self.marks[asset_index],
                       s=150,
                       color='black',
                       label=self.risky_assets[asset_index])
        ax.scatter(x=self.max_sharpe_portf.volatility,
                   y=self.max_sharpe_portf.returns,
                   c='black', marker='*',
                   s=200, label = 'Max Sharpe Ratio')
        ax.scatter(x=self.min_vol_portf.volatility,
                   y=self.min_vol_portf.returns,
                   c='black', marker='P',
                   s=200, label='Minimum Volatility')
        ax.legend()
        plt.show()
    
    def sharpe_max(self):
        self.max_sharpe_ind = np.argmax(self.portf_results_df.sharpe_ratio)
        self.max_sharpe_portf = self.portf_results_df.loc[self.max_sharpe_ind]
        
        self.min_vol_ind = np.argmin(self.portf_results_df.volatility)
        self.min_vol_portf = self.portf_results_df.loc[self.min_vol_ind]
        
        
        print('Maximum Sharpe Ratio portfolio ----')
        print('Performance')
        for index, value in self.max_sharpe_portf.items():
            print(f'{index}: {100 * value:.2f}% ', end="", flush=True)
        print('\nWeights')
        for x, y in zip(self.risky_assets, self.weights[np.argmax(self.portf_results_df.sharpe_ratio)]):
            print(f'{x}: {100*y:.2f}% ', end="", flush=True)
        
        
        print('\n\nMinimum Volatility portfolio ----')
        print('Performance')
        for index, value in self.min_vol_portf.items():
            print(f'{index}: {100 * value:.2f}% ', end="", flush=True)
        print('\nWeights')
        for x, y in zip(self.risky_assets, self.weights[np.argmin(self.portf_results_df.volatility)]):
            print(f'{x}: {100*y:.2f}% ', end="", flush=True)
#%% Efficient Frontier SciPy  
    def Get_Efficient_Frontier_SciPy(self):
        self.efficient_portfolios = []
        self.n_assets= len(self.avg_returns)
        self.args = (self.avg_returns, self.cov_mat)
        self.bounds = tuple((0,1) for asset in range(self.n_assets))
        initial_guess = self.n_assets * [1. / self.n_assets,]
        for ret in self.rtns_range:
            constraints = ({'type': 'eq',
                            'fun': lambda x: self.get_portf_rtn(x, self.avg_returns) - ret},
                           {'type': 'eq',
                            'fun': lambda x: np.sum(x) - 1})
            efficient_portfolio = sco.minimize(self.get_portf_vol, initial_guess, args=self.args,
                                               method = 'SLSQP', constraints=constraints, 
                                               bounds = self.bounds)
            self.efficient_portfolios.append(efficient_portfolio)
        return self. efficient_portfolios
            
    def get_portf_rtn(self, w, avg_rtns):
        return np.sum(avg_rtns * w)
    
    def get_portf_vol(self, w, avg_rtns, 
                      cov_mat):
        return np.sqrt(np.dot(w.T, np.dot(cov_mat, w)))
    
    def Main_SciPy(self):
        self.data()
        self.portfolio_metrics()
        self.rtns_range = np.linspace(-0.22, 0.32, 200)
        self.Get_Efficient_Frontier_SciPy()
        self.vols_range = [x['fun'] for x in self.efficient_portfolios]
        #plotting
        fig, ax = plt.subplots()
        self.portf_results_df.plot(kind='scatter', x='volatility', y='returns',
                              c='sharpe_ratio', cmap='RdYlGn', edgecolors='black',
                              ax=ax)
        ax.plot(self.vols_range, self.rtns_range, 'b--', linewidth = 3)
        ax.set(xlabel='Volatility',
               ylabel='Expected Returns',
               title='Efficient Frontier')
        plt.show()
        #minimum volatility 
        self.min_vol_ind = np.argmin(self.vols_range)
        self.min_vol_portf_rtn = self.rtns_range[self.min_vol_ind]
        self.min_vol_portf_vol = self.efficient_portfolios[self.min_vol_ind]['fun']
        
        self.min_vol_portf = {'Return': self.min_vol_portf_rtn,
                              'Volatility': self.min_vol_portf_vol,
                              'Sharpe Ratio': (self.min_vol_portf_rtn / self.min_vol_portf_vol)}
        #print
        print('\n\nMinimum Volatility portfolio ----')
        print('Performance')
        for index, value in self.min_vol_portf.items():
            print(f'{index}: {100 * value:.2f}% ', end="", flush=True)
        print('\nWeights')
        for x, y in zip(self.risky_assets, self.efficient_portfolios[self.min_vol_ind]['x']):
            print(f'{x}: {100*y:.2f}% ', end="", flush=True)
        
        self.find_opt_port()
        
    #optimizing for negative share ratio
    def neg_sharpe_ratio(self, w, avg_rtns, cov_mat, rf_rate):
        
        portf_returns = np.sum(avg_rtns * w)
        portf_volatility = np.sqrt(np.dot(w.T, np.dot(cov_mat, w)))
        portf_sharpe_ratio = (portf_returns - rf_rate) / portf_volatility 
        return -portf_sharpe_ratio
    
    def find_opt_port(self):
        self.n_assets = len(self.avg_returns)
        self.rf_rate = 0
        
        self.args = (self.avg_returns, self.cov_mat, self.rf_rate)
        self.constraints = ({'type': 'eq',
                        'fun': lambda x: np.sum(x) - 1})
        self.bounds = tuple((0, 1) for asset in range(self.n_assets))
        self.initial_guess = self.n_assets * [1. / self.n_assets]
        

        self.max_sharpe_portf = sco.minimize(self.neg_sharpe_ratio,
                                        x0 = self.initial_guess,
                                        args=self.args,
                                        method='SLSQP',
                                        bounds=self.bounds,
                                        constraints=self.constraints)

        self.max_sharpe_portf_w = self.max_sharpe_portf['x']
        self.max_sharpe_portf = {'Return': self.get_portf_rtn(self.max_sharpe_portf_w,
                                                         self.avg_returns),
                            'Volatility': self.get_portf_vol(self.max_sharpe_portf_w,
                                                             self.avg_returns,
                                                             self.cov_mat),
                            'Sharpe Ratio': -self.max_sharpe_portf['fun']}
        print('\nMaximum Sharpe Ratio portfolio ----')
        print('Performance')
        for index, value in self.max_sharpe_portf.items():
            print(f'{index}: {100 * value:.2f}% ', end="", flush=True)
        print('\nWeights')
        for x, y in zip(self.risky_assets, self.max_sharpe_portf_w):
            print(f'{x}: {100*y:.2f}% ', end="", flush=True)
        
#%%
#%% Efficient frontier with convex optimalization 
    def Get_Efficient_Frontier_CP(self):
        self.data()
        #self.portfolio_metrics()
        
        self.avg_returns = self.avg_returns.values
        self.cov_mat = self.cov_mat.values
        
        #optimization problem
        self.weights = cp.Variable(self.n_assets)
        self.gamma = cp.Parameter(nonneg=True)
        self.portf_rtn_cvx = self.avg_returns @ self.weights
        self.portf_vol_cvx = cp.quad_form(self.weights, self.cov_mat)
        self.objective_function = cp.Maximize(self.portf_rtn_cvx - self.gamma * \
                                              self.portf_vol_cvx)
        self.problem = cp.Problem(self.objective_function, [cp.sum(self.weights) == 1,
                                                            self.weights >= 0])
        
        #Calculate Efficient Frontier 
        self.n_points = 25
        self.portf_rtn_cvx_ef = np.zeros(self.n_points)
        self.portf_vol_cvx_ef = np.zeros(self.n_points)
        self.weights_ef = []
        self.gamma_range = np.logspace(-3, 3, num=self.n_points)
        
        for i in range(self.n_points):
            self.gamma.value = self.gamma_range[i]
            self.problem.solve()
            self.portf_vol_cvx_ef[i] = cp.sqrt(self.portf_vol_cvx).value
            self.portf_rtn_cvx_ef[i] = self.portf_rtn_cvx.value
            self.weights_ef.append(self.weights.value)
        #plotting risk averseness
        self.weights_df = pd.DataFrame(self.weights_ef,
                                       columns=self.risky_assets,
                                       index=np.round(self.gamma_range, 3))
        ax = self.weights_df.plot(kind='bar', stacked = True)
        ax.set(title='Weights allocation per risk-aversion level',
               xlabel=r'$\gamma$',
               ylabel='weight')
        ax.legend(bbox_to_anchor=(1,1))
        #plotting efficient frontier
        fig, ax = plt.subplots()
        ax.plot(self.portf_vol_cvx_ef, self.portf_rtn_cvx_ef, 'g--')
        for asset_index in range(self.n_assets):
            plt.scatter(x=np.sqrt(self.cov_mat[asset_index, asset_index]),
                        y=self.avg_returns[asset_index],
                        marker=self.marks[asset_index],
                        label=self.risky_assets[asset_index],
                        s=150)
        ax.set(title='Efficient Frontier',
               xlabel='Volatility',
               ylabel='Expected Returns')
        ax.legend()
        plt.show()
        
    def compare(self):
        self.Main_SciPy()
        self.Get_Efficient_Frontier_CP()
        
        self.x_lim = [0.25, 0.6]
        self.y_lim = [0.125, 0.325]
        
        fig, ax = plt.subplots(1, 2)
        ax[0].plot(self.vols_range, self.rtns_range, 'g-', linewidth=3)
        ax[0].set(title='Efficient Frontier - Minimized Volatility',
                  xlabel = 'Volatility',
                  ylabel = 'Expected Returns',
                  xlim = self.x_lim,
                  ylim = self.y_lim)
        ax[1].plot(self.portf_vol_cvx_ef, self.portf_rtn_cvx_ef, 'g-', linewidth=3)
        ax[1].set(title='Efficient Frontier - Maximized Risk-Adjusted Return',
                  xlabel = 'Volatility',
                  ylabel= 'Expected Returns', 
                  xlim = self.x_lim,
                  ylim = self.y_lim)
        plt.show()
        
    def leverage(self):
        self.Get_Efficient_Frontier_CP()
        
        self.max_leverage = cp.Parameter()
        self.problem_with_leverage = cp.Problem(self.objective_function, 
                                                [cp.sum(self.weights) == 1,
                                                 cp.norm(self.weights, 1) <= self.max_leverage])
        
        self.leverage_range = [1,2,5]
        self.len_leverage = len(self.leverage_range)
        self.n_points = 25
        
        self.portf_vol_1_ef = np.zeros((self.n_points, self.len_leverage))
        self.portf_rtn_1_ef = np.zeros((self.n_points, self.len_leverage))
        self.weights_ef = np.zeros((self.len_leverage, self.n_points, self.n_assets))
        
        for lev_ind, leverage in enumerate(self.leverage_range):
            for gamma_ind in range(self.n_points):
                self.max_leverage.value = leverage 
                self.gamma.value = self.gamma_range[gamma_ind]
                self.problem_with_leverage.solve()
                self.portf_vol_1_ef[gamma_ind, lev_ind] = cp.sqrt(self.portf_vol_cvx).value
                self.portf_rtn_1_ef[gamma_ind, lev_ind] = self.portf_rtn_cvx.value
                self.weights_ef[lev_ind, gamma_ind, :] = self.weights.value
        
        #plotting
        fig, ax = plt.subplots()
        
        for leverage_index, leverage in enumerate(self.leverage_range):
            plt.plot(self.portf_vol_1_ef[:, leverage_index],
                     self.portf_rtn_1_ef[:, leverage_index],
                     label=f'{leverage}')
        
        ax.set(title='Efficient Frontier for different max leverage',
               xlabel = 'Volatility',
               ylabel = 'Expected Returns')
        ax.legend(title='Max Leverage')
        plt.show()
        #Weight allocation
        fig, ax = plt.subplots(self.len_leverage, 1, sharex = True)
        
        for ax_index in range(self.len_leverage):
            weights_df = pd.DataFrame(self.weights_ef[ax_index],
                                      columns=self.risky_assets,
                                      index=np.round(self.gamma_range, 3))
            weights_df.plot(kind='bar', stacked=True, ax=ax[ax_index], legend=None)
            ax[ax_index].set(ylabel=(f'max_leverage = {self.leverage_range[ax_index]} \n weight'))
            
        ax[self.len_leverage - 1].set(xlabel=r'$\gamma$')
        ax[0].legend(bbox_to_anchor=(1,1))
        ax[0].set_title('Weights allocation per risk-aversion level', fontsize=16)
        plt.show()
        
        
x = Efficient_Frontier()

x.leverage()