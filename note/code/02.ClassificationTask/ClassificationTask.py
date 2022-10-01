"""
 * description: 分类任务
 * date: 2022/10/01/20:59:00
 * author: xinyu
 * version: 1.0
"""

from pathlib import Path
import requests
import pickle
import gzip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from sklearn import preprocessing
import tensorflow.keras
# 下载mnist数据集
DATA_PATH = Path('data')
PATH = DATA_PATH/"mnist"
#
PATH.mkdir(parents=True,exist_ok=True)
#
# URL = "http://deeplearning.net/data/mnist/"
FILENAME = "mnist.pkl.gz"
#
# if not (PATH/FILENAME).exists():
#     content = requests.get(URL+FILENAME).content
#     (PATH/FILENAME).open('wb').write(content)

# with gzip.open((PATH/FILENAME).as_posix(),'rb') as f :
#     ((x_train,y_train),(x_valid,y_valid),_) = pickle.load(f,encoding='latin-1')

# 下载访问不了 手动下载 解包
with open(PATH/'mnist.pkl', 'rb') as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding='latin-1')


# 一定选择对应适合的损失函数
model = tf.keras.Sequential()
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))

# compile相当于对网络进行配置，指定好优化器和损失函数等
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),loss=tf.keras.losses.SparseCategoricalCrossentropy(),metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

model.fit(x_train,y_train,epochs=5,batch_size=64,validation_data=(x_valid,y_valid))