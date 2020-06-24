#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 19:47:10 2020

@author: Max
"""

from datetime import datetime
import backtrader as bt 

#define class representing trading strategy
class RsiSignalStrategy(bt.SignalStrategy):
    params = dict(rsi_periods=14, rsi_upper=70,
                  rsi_lower=30, rsi_mid=50)

    def __init__(self):
        rsi = bt.indicators.RSI(period=self.p.rsi_periods,
                                upperband=self.p.rsi_upper,
                                lowerband=self.p.rsi_lower,
                                plot = False)
        
        bt.talib.RSI(self.data, plotname='TA_RSI')
        rsi_signal_long = bt.ind.CrossUp(rsi, self.p.rsi_lower,
                                         plot = False)
        self.signal_add(bt.SIGNAL_LONG, rsi_signal_long)
        self.signal_add(bt.SIGNAL_LONGEXIT, -(rsi > self.p.rsi_mid))
        
        rsi_signal_short = -bt.ind.CrossDown(rsi, self.p.rsi_upper,
                                             plot=False)
        self.signal_add(bt.SIGNAL_SHORT, rsi_signal_short)
        self.signal_add(bt.SIGNAL_SHORTEXIT, rsi < self.p.rsi_mid)

#data
data = bt.feeds.YahooFinanceData(dataname='FB',
                                 fromdate = datetime(2018,1,1),
                                 todate = datetime(2018,12,31)
                                 )
#set-up
cerebro = bt.Cerebro(stdstats = False)

cerebro.addstrategy(RsiSignalStrategy)
cerebro.adddata(data)
cerebro.broker.setcash(1000.0)
cerebro.broker.setcommission(commission=0.001)
cerebro.addobserver(bt.observers.BuySell)
cerebro.addobserver(bt.observers.Value)


#Run
cerebro.run()


cerebro.plot(iplot=True, volume=False)