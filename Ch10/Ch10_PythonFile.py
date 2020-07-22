#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 21:37:35 2020

@author: Max
"""



import matplotlib.pyplot as plt
import warnings

plt.style.use('seaborn')
warnings.simplefilter(action='ignore', category=FutureWarning)

# Deep learning for tabular data 

from fastai import *
from fastai.tabular import *
import torch
import pandas as pd

from chapter_10_utils import performance_evaluation_report

#Load the dataset from CSV file
df = pd.read_csv('credit_card_default.csv', index_col=0, na_values='')

df.payment_status_sep.unique()

#Identify the dependent variabe and numerical/categorical features
der_var = 'default_payment_next_month'

num_features = list(df.select_dtypes('number').columns)
num_features.remove(der_var)
cat_features = list(df.select_dtypes('object').columns)

preprocessing = [FillMissing, Categorify, Normalize]

#Create TabularDataBunch from the DataFrame
data = (TabularList.from_df(df,
                           cat_names=cat_features, 
                           cont_names = num_features,
                           procs = preprocessing)
                   .split_by_rand_pct(valid_pct=0.2, seed=42)
                   .label_from_df(cols=der_var)
                   .databunch(num_workers=0))

#Inspect rows:
data.show_batch(rows=5)

#Define learner object
learn = tabular_learner(data, layers=[1000,500],
                        ps=[0.001, 0.01],
                        emb_drop=0.04,
                        metrics=[Recall(),
                                 FBeta(beta=1),
                                 FBeta(beta=5)])

learn.model

#Find suggested learning rate:
learn.lr_find()
learn.recorder.plot(suggestion=True)
plt.show()

input = float(  input("Learning rate is: ")) 

#Train neural network
learn.fit(epochs=25, lr=input, wd=0.2)

#Extract predictions for the validation set
preds_valid, _ = learn.get_preds(ds_type=DatasetType.Valid)
pred_valid = preds_valid.argmax(dim=-1)


#Inspect the performance 
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()

#Inspect performance evaluation metrics 
performance_evaluation_report(learn)