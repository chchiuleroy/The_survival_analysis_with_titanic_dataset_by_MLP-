# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 16:32:07 2017

@author: leroy
"""

import urllib.request
import os

url = "http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic3.xls"
filepath = "datas/titanic3.xls"
if not os.path.isfile(filepath):
    results = urllib.request.urlretrieve(url, filepath)
    print("download:", results)

import numpy as np
import pandas as pd

df = pd.read_excel(filepath)
df[:2]

cols = ["survived", "pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
df = df[cols]
df.isnull().sum()

age_mean = df["age"].mean()
df["age"] = df["age"].fillna(age_mean)

fare_mean = df["fare"].mean()
df["fare"] = df["fare"].fillna(fare_mean)

df["sex"] = df["sex"].map({"female":0, "male":1}).astype("int")

df_n = pd.get_dummies(data = df, columns = ["embarked"])
df_n[:3]

df_ar = df_n.values
df_ar.shape
df_ar[:3]

label = df_ar[:, 0]
factor = df_ar[:, 1:]

label[:3]
factor[:3]

from sklearn import preprocessing as pp

minmax_scale = pp.MinMaxScaler(feature_range = (0, 1))
factor_s = minmax_scale.fit_transform(factor)

factor_s[:3]

ind = np.random.rand(len(label))<0.8
train_label = label[ind]
test_label = label[~ind]

train_factor = factor_s[ind]
test_factor = factor_s[~ind]

from keras.models import Sequential as seq
from keras.layers.core import Dense, Dropout, Activation

model = seq()
model.add(Dense(units = 30, input_dim = 9, kernel_initializer = "uniform", activation = "relu"))
model.add(Dropout(0.3))
model.add(Dense(units = 10, kernel_initializer = "uniform", activation = "relu"))
model.add(Dropout(0.3))
model.add(Dense(units = 1, kernel_initializer = "uniform", activation = "sigmoid"))
model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])

train_history = model.fit(train_factor, train_label, epochs = 20, batch_size = 50, validation_split = 0.2, verbose = 2)
scores = model.evaluate(test_factor, test_label)
scores[1]