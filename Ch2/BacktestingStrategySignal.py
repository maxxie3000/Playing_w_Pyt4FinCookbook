#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 19:47:10 2020

@author: Max
"""

from datetime import datetime
import backtrader as bt 

#define class representing trading strategy
class SmaSignal(bt.Signal):
    params = (('period', 20), )
    def __init__(self):
        self.lines.signal = self.data - bt.ind.SMA(period=self.p.period)
#data
data = bt.feeds.YahooFinanceData(dataname='AAPL',
                                 fromdate = datetime(2018,1,1),
                                 todate = datetime(2018,12,31)
                                 )
#set-up
cerebro = bt.Cerebro(stdstats = False)

cerebro.adddata(data)
cerebro.broker.setcash(1000.0)
cerebro.add_signal(bt.SIGNAL_LONG, SmaSignal)
cerebro.addobserver(bt.observers.BuySell)
cerebro.addobserver(bt.observers.Value)


#Run
print(f'Starting Portfolio Value: {cerebro.broker.getvalue():.2f}')
cerebro.run()
print(f'Final Portfolio Value: {cerebro.broker.getvalue():.2f}')

cerebro.plot(iplot=True, volume=False)