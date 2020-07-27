#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 20:15:59 2020

@author: Max
"""

#%% import libraries 
import matplotlib.pyplot as plt

import yfinance as yf
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

from torch.utils.data import (Dataset, TensorDataset, 
                              DataLoader, Subset)

from chapter_10_utils import create_input_data, custom_set_seed

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler 

device = 'cpu'

#%% Define parameters
# Data
ticker = 'INTL'
start_date = '2010-01-02'
end_date = '2019-12-31'
valid_start = '2019-07-01'
n_lags = 12

# Neural network 
batch_size = 16
n_epochs = 100

#%% Download and prepare the data
df = yf.download(ticker,
                 start=start_date,
                 end=end_date,
                 progress=False)

df = df.resample("W-MON").last()
valid_size = df.loc[valid_start:end_date].shape[0]
prices = df['Adj Close'].values.reshape(-1, 1)

#%% Scale the time series of prices 
valid_ind = len(prices) - valid_size 

minmax = MinMaxScaler(feature_range=(0, 1))

prices_train = prices[:valid_ind]
prices_valid = prices[valid_ind:]

minmax.fit(prices_train)

prices_train = minmax.transform(prices_train)
prices_valid = minmax.transform(prices_valid)

prices_scaled = np.concatenate((prices_train, prices_valid)).flatten()

#%% Transform the time series into input for the RNN
X, y = create_input_data(prices_scaled, n_lags)

#%% Obtain Naive forecast
naive_pred = prices[len(prices) - valid_size - 1:-1]
y_valid = prices[len(prices) - valid_size:]

naive_mse = mean_squared_error(y_valid, naive_pred)
naive_rmse = np.sqrt(naive_mse)
print(f"Naive forecast - MSE: {naive_mse:.4f}, RMSE: {naive_rmse:.4f}")

#%% Prepare the DataLoader objects 
#set seed for reproducibility 
custom_set_seed(42)

valid_ind = len(X) - valid_size

X_tensor = torch.from_numpy(X).float().reshape(X.shape[0],
                                               X.shape[1],
                                               1)
y_tensor = torch.from_numpy(y).float().reshape(X.shape[0], 1)

dataset = TensorDataset(X_tensor, y_tensor)

train_dataset = Subset(dataset, list(range(valid_ind)))
valid_dataset = Subset(dataset, list(range(valid_ind, len(X))))

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True)
valid_loader = DataLoader(dataset=valid_dataset,
                          batch_size=batch_size)

#%% Define the model
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, output_size):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size,
                          n_layers, batch_first=True,
                          nonlinearity='relu')
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        output, _ = self.rnn(x)
        output = self.fc(output[:, -1,:])
        return output 
    
#%% Instantiate the model, loss function and optimizer
model = RNN(input_size=1, hidden_size=6, 
            n_layers=1, output_size=1).to(device)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#%% Train the network 
print_every = 10 
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
            y_val = y_val.to(device)

            y_hat = model(x_val)
            loss = torch.sqrt(loss_fn(y_val, y_hat))
            running_loss_valid += loss.item() * x_val.size(0)
        epoch_loss_valid = running_loss_valid / len(valid_loader.dataset)
        if epoch > 0 and epoch_loss_valid < min(valid_losses):
            best_epoch = epoch
            torch.save(model.state_dict(), './rnn_checkpoint.pth')
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
state_dict = torch.load('rnn_checkpoint.pth')
model.load_state_dict(state_dict)

#%% Obtain predicitions
y_pred = []

with torch.no_grad():
    model.eval()
    for x_val, y_val in valid_loader:
        x_val = x_val.to(device)
        y_hat = model(x_val)
        y_pred.append(y_hat)

y_pred = torch.cat(y_pred).numpy()
y_pred = minmax.inverse_transform(y_pred).flatten()

#%% Evaluate predictions
mlp_mse = mean_squared_error(y_valid, y_pred)
mlp_rmse = np.sqrt(mlp_mse)
print(f"RNN's forecast - MSE: {mlp_mse:.2f}, RMSE: {mlp_rmse:.2f}")

fig, ax = plt.subplots()

ax.plot(y_valid, color='blue', label='True')
ax.plot(y_pred, color='red', label='Prediction')
ax.plot(naive_pred, color='green', label='Naive')

ax.set(title="RNN's Forecasts",
       xlabel="Time",
       ylabel='Price ($)')
ax.legend()

    
    
        