#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 19:47:10 2020

Template:
    
    class SmaSignal(bt.Strategy):
    
        def __init__(self):
            #some code
            
        def log (self, txt):
            #some code
            
        def notify_order(self, order):
            #some code
        
        def notify_trade(self, trade):
        
        def next(self):
            #some code

@author: Max
"""

import datetime
import backtrader as bt
import pandas as pd 
import matplotlib.pyplot as plt

#define class representing trading strategy
class BBand_Strategy(bt.Strategy):
    params = (('period', 20),
              ('devfactor', 2.0),)
    
    def __init__(self):
        
        # keep track of close price in the series
        self.data_close = self.datas[0].close
        self.data_open = self.datas[0].open
        
        # keep track of pending orders/buy price/buy commission
        self.order = None
        self.price = None
        self.comm = None

        # add Bolling bands
        self.b_band = bt.ind.BollingerBands(self.datas[0],
                                            period=self.p.period,
                                            devfactor=self.p.devfactor)
        
        self.buy_signal = bt.ind.CrossOver(self.datas[0],
                                           self.b_band.lines.bot)
        self.sell_signal = bt.ind.CrossOver(self.datas[0],
                                            self.b_band.lines.top)
        
    def log (self, txt):
        dt = self.datas[0].datetime.date(0).isoformat()
        print(f'{dt}, {txt}')
        
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED --- Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Commission: {order.executed.comm:.2f}')
                self.price = order.executed.price
                self.comm = order.executed.comm
            else:
                self.log(f'SELL EXECUTED --- Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Commission: {order.executed.comm:.2f}')
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Failed')
        
        self.order = None
                
    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        
        self.log(f'OPERATION RESULT --- Gross: {trade.pnl:.2f}, Net: {trade.pnlcomm:.2f}')
    
    def next_open(self):
        if not self.position:
            if self.buy_signal > 0:
                size = int(self.broker.getcash() / self.datas[0].open)
                self.log(f'BUY CREATED --- Size: {size}, Cash: {self.broker.getcash():.2f}, Open: {self.data_open[0]}, Close: {self.data_close[0]}')
                self.buy(size=size)
        else:
            if self.sell_signal < 0:
                self.log(f'SELL CREATED --- Size: {self.position.size}')
                self.sell(size=self.position.size)
   # def stop(self):
        
            
    
#data
data = bt.feeds.YahooFinanceData(dataname='WMT',
                                 fromdate = datetime.datetime(2018,1,1),
                                 todate = datetime.datetime(2018,12,31)
                                 )
#set-up
cerebro = bt.Cerebro(stdstats = False, cheat_on_open=True)
cerebro.addstrategy(BBand_Strategy)
cerebro.adddata(data)
cerebro.broker.setcash(10000.0)
cerebro.broker.setcommission(commission=0.001)
#Optimize strategy
#cerebro.optstrategy(SmaStrategy, ma_period = range(10,31))
cerebro.addobserver(bt.observers.BuySell)
cerebro.addobserver(bt.observers.Value)
cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='time_return')



#Run
print(f'Starting Portfolio Value: {cerebro.broker.getvalue():.2f}')
backtest_result = cerebro.run()
print(f'Final Portfolio Value: {cerebro.broker.getvalue():.2f}')

print(backtest_result[0].analyzers.returns.get_analysis())

cerebro.plot(iplot=True, volume=False)

returns_dict = backtest_result[0].analyzers.time_return.get_analysis()
returns_df = pd.DataFrame(list(returns_dict.items()), columns = ['report_date', 'return']) \
                .set_index('report_date')

returns_df.plot(title='Portfolio returns')
plt.tight_layout()
plt.show()
