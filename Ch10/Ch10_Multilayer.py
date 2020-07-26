#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 21:46:39 2020

Estimate a multilayer perceptron for financial time series forecasting
(Example, do not use for "real" predicitions)

@author: Max
"""

import matplotlib.pyplot as plt

import yfinance as yf
import numpy as np

import torch 
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import (Dataset, TensorDataset,
                              DataLoader, Subset)

from sklearn.metrics import mean_squared_error

device = 'cpu' #No GPU cuda on MacBook 

#%% Define the parameters

#Data
ticker = 'ANF'
start_date = '2000-01-02'
end_date = '2019-12-31'
n_lags = 8

#Neural network
valid_size = 12
batch_size = 5
n_epochs = 10 ** 3

#%% Download Stock prices and process the data 
df = yf.download(ticker, 
                start = start_date,
                end = end_date,
                progress=False)

df = df.resample("M").last()
prices = df['Adj Close'].values 

#%% Define a function for transforming time series into a dataset for the MLP (NN)
def create_input_data(series, n_lags=1):
    X, y = [], []
    for step in range(len(series) - n_lags):
        end_step = step + n_lags
        X.append(series[step:end_step])
        y.append(series[end_step])
    return np.array(X), np.array(y)

#%% Transform the considered time series into input for the MLP
X, y = create_input_data(prices, n_lags)

#Tranform to tensor
X_tensor = torch.from_numpy(X).float()
y_tensor = torch.from_numpy(y).float().unsqueeze(dim=1)

# X_tensor.type() --> Can also be used to see where tensor is located (cuda: GPU)

#%% Create training and validation sets
valid_ind = len(X) - valid_size

dataset = TensorDataset(X_tensor, y_tensor)

train_dataset = Subset(dataset, list(range(valid_ind)))
valid_dataset = Subset(dataset, list(range(valid_ind, len(X))))

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size)
valid_loader = DataLoader(dataset=valid_dataset,
                          batch_size=batch_size)

#%% Inspecting
# print(next(iter(train_loader))[0])
# print(next(iter(train_loader))[1])

#%% Usa a naïve forecast as a benchmark and evaluate the performance
naive_pred = prices[len(prices) - valid_size - 1: -1]
y_valid = prices[len(prices) - valid_size:]

naive_mse = mean_squared_error(y_valid, naive_pred)
naive_rmse = np.sqrt(naive_mse)
print(f"Naive forecast - MSE: {naive_mse:.2f}, RMSE: {naive_rmse:.2f}")

#%% Define the network's architecture 
class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(input_size, 8)
        self.linear2 = nn.Linear(8,4)
        self.linear3 = nn.Linear(4,1)
        self.dropout = nn.Dropout(p=0.2)
    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear3(x)
        return x

#instiate the model, the loss function and the optimizer 
#Set seed for reproducability
torch.manual_seed(42)

model = MLP(n_lags).to(device)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

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
        y_batch = y_batch.to(device)
        y_hat = model(x_batch) 
        loss = loss_fn(y_batch, y_hat)
        loss.backward()
        
        optimizer.step()
        running_loss_train += loss.item() * x_batch.size(0)
    epoch_loss_train = running_loss_train / len(train_loader.dataset)
    train_losses.append(epoch_loss_train)
    
    with torch.no_grad():
        model.eval()
        for x_val, y_val in valid_loader:
            x_val = x_val.to(device)
            y_val = y_val.to(device)
            y_hat = model(x_val)
            loss = loss_fn(y_val, y_hat)
            running_loss_valid += loss.item() * x_val.size(0)
        epoch_loss_valid = running_loss_valid / len(valid_loader.dataset)
        if epoch > 0 and epoch_loss_valid < min(valid_losses):
            best_epoch = epoch
            torch.save(model.state_dict(), './mlp.checkpoint.pth')
        valid_losses.append(epoch_loss_valid)
    
    if epoch % print_every == 0:
        print(f"<{epoch}> - Train. loss: {epoch_loss_train:.2f} \t Valid. loss: {epoch_loss_valid:.2f}")
print(f'Lowest loss recorded in epoch: {best_epoch}')

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

#%% Load the best model
state_dict = torch.load('mlp.checkpoint.pth')
model.load_state_dict(state_dict)

#%% Obtain predicitions
y_pred, y_valid = [], []

with torch.no_grad():
    model.eval()
    for x_val, y_val in valid_loader:
        x_val = x_val.to(device)
        y_pred.append(model(x_val))
        y_valid.append(y_val)
y_pred = torch.cat(y_pred).numpy().flatten()
y_valid = torch.cat(y_valid).numpy().flatten()

#%% Evaluate predictions
mlp_mse = mean_squared_error(y_valid, y_pred)
mlp_rmse = np.sqrt(mlp_mse)
print(f"MLP's forecast - MSE: {mlp_mse:.2f}, RMSE: {mlp_rmse:.2f}")

fig, ax = plt.subplots()

ax.plot(y_valid, color='blue', label='true')
ax.plot(y_pred, color='red', label='prediction')

ax.set(title="Multilayer Perceptron's Forecasts",
       xlabel="Time",
       ylabel='Price ($)')
ax.legend()

#%% Testing

#print(model) -> provides architecture 
#print(model.dict()) --> provides weights and biases 

