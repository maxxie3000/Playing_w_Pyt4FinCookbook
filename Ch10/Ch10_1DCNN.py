#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 12:41:13 2020

Train a 1D CNN in PyTorch 

@author: Max
"""

#%% Import libraries 
import matplotlib.pyplot as plt

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

# print(model) --> control architecture

#%% Instantiate the model, the loss function and the optimizer
model = model.to(device)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

#%% Train the network
print_every = 50
train_losses, valid_losses = [], []

for epoch in range(n_epochs):
    running_loss_train = 0
    running_loss_valid = 0
    
    model.train()
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        x_batch = x_batch.to(device)
        x_batch = x_batch.view(x_batch.shape[0], 1 , n_lags)
        
        y_batch = y_batch.to(device)
        y_batch = y_batch.view(y_batch.shape[0], 1, 1)
        y_hat = model(x_batch).view(y_batch.shape[0], 1, 1)
        
        loss = torch.sqrt(loss_fn(y_batch, y_hat))
        loss.backward()
        optimizer.step()
        running_loss_train += loss.item() * x_batch.size(0)
    epoch_loss_train = running_loss_train / len(train_loader.dataset)
    train_losses.append(epoch_loss_train)
    
    with torch.no_grad():
        model.eval()
        for x_val, y_val in valid_loader:
            x_val = x_val.to(device)
            x_val = x_val.view(x_val.shape[0], 1, n_lags)
            
            y_val = y_val.to(device)
            y_val = y_val.view(y_val.shape[0], 1, 1)
            
            y_hat = model(x_val).view(y_val.shape[0], 1, 1)
            loss = torch.sqrt(loss_fn(y_val, y_hat))
            running_loss_valid += loss.item() * x_val.size(0)
        epoch_loss_valid = running_loss_valid / len(valid_loader.dataset)
        if epoch > 0 and epoch_loss_valid < min(valid_losses):
            best_epoch = epoch
            torch.save(model.state_dict(), './cnn_checkpoint.pth')
        valid_losses.append(epoch_loss_valid)
        
    if epoch % print_every == 0:
        print(f'<{epoch}> - Train. loss: {epoch_loss_train:.6f} \t Valid. Loss: {epoch_loss_valid:.6f}')
print(f"Lowest loss recorded in epoch: {best_epoch}")

#%% Plot the losses over epochs
train_losses = np.array(train_losses)
valid_losses = np.array(valid_losses)

fig, ax = plt.subplots()

ax.plot(train_losses, color='blue', label='Training loss')
ax.plot(valid_losses, color='red', label='Validation loss')

ax.set(title='loss over epochs',
       xlabel='Epoch',
       ylabel='Loss')
ax.legend()
plt.show()

#%% Load the best model (Validation loss)
state_dict = torch.load('cnn_checkpoint.pth')
model.load_state_dict(state_dict)

#%% Obtain predicitions
y_pred, y_valid = [], []

with torch.no_grad():
    model.eval()
    for x_val, y_val in valid_loader:
        x_val = x_val.to(device)
        x_val = x_val.view(x_val.shape[0], 1, n_lags)
        y_pred.append(model(x_val))
        y_valid.append(y_val)
y_pred = torch.cat(y_pred).numpy().flatten()
y_valid = torch.cat(y_valid).numpy().flatten()

#%% Evaluate predictions
mlp_mse = mean_squared_error(y_valid, y_pred)
mlp_rmse = np.sqrt(mlp_mse)
print(f"CNN's forecast - MSE: {mlp_mse:.2f}, RMSE: {mlp_rmse:.2f}")

fig, ax = plt.subplots()

ax.plot(y_valid, color='blue', label='true')
ax.plot(y_pred, color='red', label='prediction')

ax.set(title="Multilayer Perceptron's Forecasts",
       xlabel="Time",
       ylabel='Price ($)')
ax.legend()

    
    