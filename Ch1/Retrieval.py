#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 14:38:41 2020

@author: Max
"""



import numpy as np 
import pandas as pd
import yfinance as yf
import quandl  

#Yahoo
df_yahoo = yf.download(['AAPL','MSFT'],actions = 'inline' ,progress=True)
print(df_yahoo)

#Intrinsio set-up
import intrinio_sdk

intrinio_sdk.ApiClient().configuration.api_key['api_key'] = "OjZkMThlZjVjOWUyMjk1ZjBmMDhjNGVjZTRhOTYzYjUw"
security_api = intrinio_sdk.SecurityApi()
#Intrinsio call 
r = security_api.get_security_stock_prices(identifier='AAPL',
                                           start_date='2000-01-01',
                                           end_date='2010-12-31',
                                           frequency='daily',
                                           page_size=1000
                                           )
response_list = [x.to_dict() for x in r.stock_prices]
df_intrinio = pd.DataFrame(response_list).sort_values('date')
df_intrinio.set_index('date', inplace = True)
print(df_intrinio)





