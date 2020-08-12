#Reference: https://machinelearningmastery.com/time-series-prediction-with-deep-learning-in-python-with-keras/
import os
import urllib.request
from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename
import csv, sqlite3
import jinja2
import json
import datetime
import codecs

import pandas_datareader as pdr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import array

from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

import math
from sklearn.metrics import mean_squared_error

app = Flask(__name__)
app.secret_key = os.urandom(24)

@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/', methods = ['POST'])
def predict_stock():
    ticker = request.form['ticker']
    key = '014a4686e18062db68a468ebb394827a364439e6'
    try:
        df = pdr.get_data_tiingo(ticker, api_key = key)
    except:
        flash("Not a valid ticker!")
        return redirect(request.url)

    df_extra = df.reset_index()['close']
    scaler = MinMaxScaler(feature_range = (0, 1))
    df_extra = scaler.fit_transform(np.array(df_extra).reshape(-1, 1))

    train_size = int(len(df_extra) * 0.65)
    test_size = len(df_extra) - train_size
    train_data, test_data = df_extra[0:train_size, :], df_extra[train_size:len(df_extra), :1]

    def dataset(dataset, time_step = 1):
        dataX, dataY = [], []
        for i in range(len(dataset) - time_step - 1):
            e = dataset[i:(i + time_step), 0] 
            dataX.append(e)
            dataY.append(dataset[i + time_step, 0])
        return np.array(dataX), np.array(dataY)

    time_step = 100
    x_train, y_train = dataset(train_data, time_step)
    x_test, y_test = dataset(test_data, time_step)
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1] , 1))
    try:
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1] , 1))
    except:
        flash("Not enough information to make an accurate prediction! Try another ticker.")
        return redirect(request.url)

    # LSTM
    model = Sequential()
    model.add(LSTM(50, return_sequences = True, input_shape = (100, 1)))
    model.add(LSTM(50, return_sequences = True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss = 'mean_squared_error', optimizer='adam')
    model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = 100, batch_size = 32, verbose = 1)

    train_predict = model.predict(x_train)
    test_predict = model.predict(x_test)

    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)

    x_input = test_data[340:].reshape(1,-1)

    temp_input = list(x_input)
    temp_input = temp_input[0].tolist()

    lst = []
    steps=100
    i=0

    for i in range(30):
        if(len(temp_input) > 100):
            x_input = np.array(temp_input[1:])
            x_input = x_input.reshape(1, -1)
            x_input = x_input.reshape((1, steps, 1))
            yhat_index = model.predict(x_input, verbose = 0)
            temp_input.extend(yhat_index[0].tolist())
            temp_input = temp_input[1:]
            lst.extend(yhat_index.tolist())
        else:
            x_input = x_input.reshape((1, steps, 1))
            yhat_index = model.predict(x_input, verbose = 0)
            temp_input.extend(yhat_index[0].tolist())
            lst.extend(yhat_index.tolist())

    df_final = df_extra.tolist()
    df_final.extend(lst)

    plt.switch_backend('Agg')
    plt.plot(df_final[1200:])

    image_add = "static/images/" + ticker + '.png'
    plt.savefig(image_add)

    return render_template('upload.html', name = ticker, url = image_add)

if __name__ == "__main__":
    app.run()

