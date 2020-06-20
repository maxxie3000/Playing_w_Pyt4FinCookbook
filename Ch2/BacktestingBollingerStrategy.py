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

from datetime import datetime
import backtrader as bt 

#define class representing trading strategy
class SmaStrategy(bt.Strategy):
    params = (('ma_period', 20), )
    def __init__(self):
        
        # keep track of close price in the series
        self.data_close = self.datas[0].close

        # keep track of pending orders/buy price/buy commission
        self.order = None
        self.price = None
        self.comm = None

        # add a simple moving average indicator
        self.sma = bt.ind.SMA(self.datas[0],
                              period = self.params.ma_period)
        
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
                
                self.bar_executed = len(self)
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Failed')
        
        self.order = None
                
    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        
        self.log(f'OPERATION RESULT --- Gross: {trade.pnl:.2f}, Net: {trade.pnlcomm:.2f}')
    
    def next(self):
        if self.order:
            return
        
        if not self.position:
            if self.data_close[0] > self.sma[0]:
                self.log(f'BUY CREATED --- Price: {self.data_close[0]:.2f}')
                self.order = self.buy(size=4)
        else:
            if self.data_close[0] < self.sma[0]:
                self.log(f'SELL CREATED --- Price: {self.data_close[0]:.2f}')
                self.order = self.sell(size=4)
    def stop(self):
        self.log(f'(ma_period = {self.params.ma_period:2d}) --- Terminal Value: {self.broker.getvalue():.2f}')
            
    
#data
data = bt.feeds.YahooFinanceData(dataname='AAPL',
                                 fromdate = datetime(2018,1,1),
                                 todate = datetime(2018,12,31)
                                 )
#set-up
cerebro = bt.Cerebro(stdstats = False)

cerebro.adddata(data)
cerebro.broker.setcash(1000.0)
#Run strategy
#cerebro.addstrategy(SmaStrategy)
#Optimize strategy
cerebro.optstrategy(SmaStrategy, ma_period = range(10,31))
cerebro.addobserver(bt.observers.BuySell)
cerebro.addobserver(bt.observers.Value)
cerebro.run(maxcpus=4)


#Run
print(f'Starting Portfolio Value: {cerebro.broker.getvalue():.2f}')
cerebro.run()
print(f'Final Portfolio Value: {cerebro.broker.getvalue():.2f}')

cerebro.plot(iplot=True, volume=False)