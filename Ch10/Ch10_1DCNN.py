#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 12:41:13 2020

Train a 1D CNN in PyTorch 

@author: Max
"""

#%% Import libraries 
import yfinance as yf 
import numpy as np
import os
import random

import torch 
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import (Dataset, TensorDataset,
                              DataLoader, Subset)
from collections import OrderedDict

from chapter_10_utils import create_input_data, custom_set_seed

from sklearn.metrics import mean_squared_error

device = 'cpu'

#%% Define the parameters 
#data
ticker = 'INTL'
start_date = '2015-01-02'
end_date = '2019-12-31'
valid_start = '2019-07-01'
n_lags = 12

#neural network
batch_size = 5
n_epochs = 2000

#%% Download and prepare the data 
df = yf.download(ticker, 
                 start=start_date,
                 end=end_date,
                 progress=False)

df = df.resample('W-MON').last()
valid_size = df.loc[valid_start:end_date].shape[0]
prices = df['Adj Close'].values

#%% Transform time series into input for the CNN 
X, y = create_input_data(prices, n_lags)

#%% Usa a naïve forecast as a benchmark and evaluate the performance
naive_pred = prices[len(prices) - valid_size - 1: -1]
y_valid = prices[len(prices) - valid_size:]

naive_mse = mean_squared_error(y_valid, naive_pred)
naive_rmse = np.sqrt(naive_mse)
print(f"Naive forecast - MSE: {naive_mse:.2f}, RMSE: {naive_rmse:.2f}")

#%% Prepare the Dataloader objects 
#Seed for reproducibility 
custom_set_seed(42)

valid_ind = len(X) - valid_size 

X_tensor = torch.from_numpy(X).float()
y_tensor = torch.from_numpy(y).float().unsqueeze(dim=1)

dataset = TensorDataset(X_tensor, y_tensor)

train_dataset = Subset(dataset, list(range(valid_ind)))
valid_dataset = Subset(dataset, list(range(valid_ind, len(X))))

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size)
valid_loader = DataLoader(dataset=valid_dataset,
                          batch_size=batch_size)

#%% Define the CNN's architecture 
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)
    
model = nn.Sequential(OrderedDict([
    ('conv_1', nn.Conv1d(1, 32, 3, padding=1)),
    ('max_pool_1', nn.MaxPool1d(2)),
    ('relu_1', nn.ReLU()),
    ('flatten', Flatten()),
    ('fc_1', nn.Linear(192, 50)),
    ('relu_2', nn.ReLU()),
    ('dropout_1', nn.Dropout(0.4)),
    ('fc_2', nn.Linear(50, 1))
]))

print(model)
    
    