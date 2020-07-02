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
import QuantLib as ql
import seaborn as sns

from scipy.stats import norm

#np.random.seed(0)

class Monte_Carlo():
    def __init__(self):
        self.risky_asset = '^AEX'
        self.risky_assets = ['GOOG', 'FB', 'MSFT']
        self.shares = [5,5,7]
        
        self.start_date = '2018-01-01'
        self.end_date = '2018-12-31'
        self.t = 1
        self.n_sim = 10 ** 5
        
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
        self.train = self.returns['2019-01-01':'2020-01-31']
        self.test = self.returns['2020-02-01':'2020-05-31']

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
        
        return self.S_t
    #European options (self-made data for simplicity)
    def EU_opt(self):
        #Data
        self.S_O = 100
        self.K = 100
        self.r = 0.05
        self.mu = self.r
        self.sigma = 0.5
        self.T = 1 #1 year
        self.N = 252 #N of days in year
        self.dt = self.T/self.N
        self.n_sim = 10 ** 6
        self.discount_factor = np.exp(-self.r * self.T)
        self.type = 'call'
        #Call black_scholes
        self.black_scholes_analytical()
        
        self.gbm_sims = self.simulate_gbm()
        self.premium = self.discount_factor * np.average(np.maximum(0, self.gbm_sims[:, -1] - self.K))
        print(self.premium)
        
    def black_scholes_analytical(self):
        
        
        self.d1 = (np.log(self.S_O/self.K) + (self.r + 0.5 * self.sigma**2) * self.T) \
                                        / (self.sigma * np.sqrt(self.T))
                           
        self.d2 = (np.log(self.S_O/self.K) + (self.r - 0.5 * self.sigma**2) * self.T) \
                                        / (self.sigma * np.sqrt(self.T))
                          
        if self.type == 'call':
            self.val = (self.S_O*norm.cdf(self.d1,0,1)-self.K*np.exp(-self.r*self.T) \
                        *norm.cdf(self.d2, 0 ,1))
        
        elif self.type == 'put':
            self.val = (self.K*np.exp(-self.r*self.T)*norm.cdf(-self.d2,0,1)-self.S_O \
                        *norm.cdf(-self.d1, 0 ,1))
        
        return self.val
        
    def US_opt(self):
        #data
        self.S_O = 36
        self.K = 40
        self.r = 0.06
        self.mu = self.r
        self.sigma = 0.2
        self.T = 1 #1 year
        self.N = 50 
        self.dt = self.T/self.N
        self.n_sim = 10 ** 5
        self.discount_factor = np.exp(-self.r * self.dt)
        self.type = 'put'
        self.poly_degree = 5
        
        
        self.gbm_sims = self.simulate_gbm()
        
        self.payoff_matrix = np.maximum(self.K - self.gbm_sims, np.zeros_like(self.gbm_sims))
        
        self.value_matrix = np.zeros_like(self.payoff_matrix)
        self.value_matrix[:,-1] = self.payoff_matrix[:,-1]
        
        for t in range(self.N - 1, 0, -1):
            self.regression = np.polyfit(self.gbm_sims[:, t],
                                         self.value_matrix[:, t + 1] * self.discount_factor,
                                         self.poly_degree)
            self.continuation_value = np.polyval(self.regression, self.gbm_sims[:, t])
            self.value_matrix[:,t] = np.where(
                self.payoff_matrix[:, t] > self.continuation_value,
                self.payoff_matrix[:, t],
                self.value_matrix[:, t + 1] * self.discount_factor)
        self.option_premium = np.mean(self.value_matrix[:,1] * self.discount_factor)
        
        print(self.option_premium)
        self.black_scholes_analytical()
        print(self.val)
        
    def US_opt_ql(self):
        self.S_O = 36
        self.K = 40
        self.r = 0.06
        self.mu = self.r
        self.sigma = 0.2
        self.T = 1 #1 year
        self.N = 50 
        self.dt = self.T/self.N
        self.n_sim = 10 ** 5
        self.discount_factor = np.exp(-self.r * self.dt)
        self.type = 'put'
        self.poly_degree = 5
        #calendar and dayconvention
        calendar = ql.UnitedStates()
        day_counter = ql.ActualActual()
        #valuation date and expiry date
        valuation_date = ql.Date(1, 1, 2018)
        expiry_date = ql.Date(1, 1, 2019)
        ql.Settings.instance().evaluationDate = valuation_date 
        #option type
        if self.type == 'call':
            option_type_ql = ql.Option.Call
        elif self.type == 'put':
            option_type_ql = ql.Option.Put
        exercise = ql.AmericanExercise(valuation_date, expiry_date)
        payoff = ql.PlainVanillaPayoff(option_type_ql, self.K)
        #prepare market-related data 
        u = ql.SimpleQuote(self.S_O)
        r = ql.SimpleQuote(self.r)
        sigma = ql.SimpleQuote(self.sigma)
        #specify market-related curves
        underlying = ql.QuoteHandle(u)
        volatility = ql.BlackConstantVol(0, ql.TARGET(),
                                         ql.QuoteHandle(sigma),
                                         day_counter)
        risk_free_rate = ql.FlatForward(0, ql.TARGET(),
                                        ql.QuoteHandle(r),
                                        day_counter)
        
        #plug market-related data into the BS process
        bs_process = ql.BlackScholesProcess(underlying, 
                                             ql.YieldTermStructureHandle(risk_free_rate),
                                             ql.BlackVolTermStructureHandle(volatility),
                                             )
        #instantiate engine
        engine = ql.MCAmericanEngine(bs_process, 'PseudoRandom',
                                     timeSteps = self.N,
                                     polynomOrder=self.poly_degree,
                                     seedCalibration = 42,
                                     requiredSamples = self.n_sim)
        #instantiate option
        option = ql.VanillaOption(payoff, exercise)
        option.setPricingEngine(engine)
        #option premium
        option_premium_ql = option.NPV()
        
        print(option_premium_ql)
        #delta
        u_0 = u.value()
        h = 0.01
        
        u.setValue(u_0 + h)
        p_plus_h = option.NPV()
        
        u.setValue(u_0 - h)
        p_minus_h = option.NPV()
        
        u.setValue(u_0)
        
        delta = (p_plus_h - p_minus_h) / (2 * h)
        print(delta)
        
   
    def VAR_MC(self):
        self.df = yf.download(self.risky_assets, start=self.start_date,
                         end=self.end_date, adjusted=True)
        self.adj_close = self.df['Adj Close']
        self.returns = self.adj_close.pct_change().dropna()
        #plot
        #plot_title = f'{" vs. ".join(self.risky_assets)} returns: {self.start_date} - {self.end_date}'
        #self.returns.plot(title=plot_title)
        #calculate covariance matrix 
        cov_mat = self.returns.cov()
        #Cholesky decomposition -> correlation (https://towardsdatascience.com/the-significance-and-applications-of-covariance-matrix-d021c17bce82)
        chol_mat = np.linalg.cholesky(cov_mat)
        #draw random and correlate
        rv = np.random.normal(size=(self.n_sim, len(self.risky_assets)))
        correlated_rv = np.transpose(np.matmul(chol_mat, np.transpose(rv)))
       #metrics for simulations
        r = np.mean(self.returns, axis=0).values
        sigma = np.std(self.returns, axis=0).values
        S_0 = self.adj_close.values[-1, :]
        P_0 = np.sum(self.shares * S_0)
        #terminal priceof stocks
        S_T = S_0 * np.exp((r - 0.5 * sigma ** 2) * self.t + sigma * np.sqrt(self.t) * correlated_rv)
        #terminal portfolio value
        P_T = np.sum(self.shares * S_T, axis = 1)
        P_diff = P_T - P_0
        #calculate VAR
        
        P_diff_sorted = np.sort(P_diff)
        percentiles = [0.01, 0.1, 1.]
        var = np.percentile(P_diff_sorted, percentiles)
        for x, y in zip(percentiles, var):
            print(f'1-day VaR with {100-x}% confidence: {-y:.2f}$')
        #plot
        ax = sns.distplot(P_diff, kde=False)
        ax.set_title('Distribution of possible 1-day changes in portfolio value 1-day 99% VaR',
                     fontsize = 16)
        ax.axvline(var[2], 0, 1)
        #conditional VaR
        var = np.percentile(P_diff_sorted, 5)
        expected_shortfall = P_diff_sorted[P_diff_sorted<=var].mean()
        print(expected_shortfall)
        
       
        
            
        
        
x = Monte_Carlo()
x.VAR_MC()
