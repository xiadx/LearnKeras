#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np


def main():
    data_dim = 16
    timesteps = 8
    num_classes = 10

    # 期望输入数据尺寸: (batch_size, timesteps, data_dim)
    model = Sequential()
    model.add(LSTM(32, return_sequences=True,
                   input_shape=(timesteps, data_dim)))  # 返回维度为 32 的向量序列
    model.add(LSTM(32, return_sequences=True))  # 返回维度为 32 的向量序列
    model.add(LSTM(32))  # 返回维度为 32 的单个向量
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    # 生成虚拟训练数据
    x_train = np.random.random((1000, timesteps, data_dim))
    y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)

    # 生成虚拟验证数据
    x_val = np.random.random((100, timesteps, data_dim))
    y_val = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)

    model.fit(x_train, y_train,
              batch_size=64, epochs=5,
              validation_data=(x_val, y_val))


if __name__ == "__main__":
    main()