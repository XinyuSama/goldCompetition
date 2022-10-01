"""
 * description: 搭建神经网络进行气温预测
 * date: 2022/10/01/15:56:00
 * author: xin yu
 * version: 1.0
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras
import warnings
import datetime
from sklearn import preprocessing
warnings.filterwarnings('ignore')

features = pd.read_csv('temps.csv')

# 数据样子
print(features)

# 处理时间数据

# 分别得到年,月,日

years = features['year']
months = features['month']
days = features['day']

# datetime格式
dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year,month,day in zip(years,months,days)]
dates = [datetime.datetime.strptime(date,'%Y-%m-%d') for date in dates]

# 准备画图
# 指定默认风格
# plt.style.use('fivethirtypeight')

# 设置布局
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2,ncols=2,figsize = (10,10)) # 创建四个画布 类似解构赋值

print(ax1,ax2)

fig.autofmt_xdate(rotation=45)
#
# 标签值
ax1.plot(dates,features['actual'])
ax1.set_xlabel(''); ax1.set_ylabel('Temperature'); ax1.set_title('Max Temp')

# 昨天
ax2.plot(dates,features['temp_1'])
ax2.set_xlabel(''); ax2.set_ylabel('Temperature'); ax2.set_title('Previous Max Temp')

# 前天
ax3.plot(dates,features['temp_2'])
ax3.set_xlabel('Date'); ax3.set_ylabel("Temperature"); ax3.set_title("Two Days Prior Max Temp")

# 我的逗逼朋友
ax4.plot(dates,features['friend'])
ax4.set_xlabel('Date'); ax4.set_ylabel("Temperature"); ax4.set_title("Friend Estimate")

plt.tight_layout(pad=2)
plt.show()

# 独热编码  处理week原来为字符串 改为数值
features = pd.get_dummies(features)
features.head(5)
print(features)

# 标签 拿到y
labels = np.array(features['actual'])

# 在特征中去掉标签 拿到x
features = features.drop('actual',axis=1)

# 名字单独保存一下，以备后患
features_list = list(features.columns)

# 转换成合适的格式
features = np.array(features)
print(features)

# 使用sklearn 做数据预处理 数据标准化
input_features = preprocessing.StandardScaler().fit_transform(features)

print(input_features)

# 按顺序构造网络模型
model = tf.keras.Sequential()
model.add(layers.Dense(16,kernel_initializer='random_normal',kernel_regularizer=tf.keras.regularizers.l2(0.03)))
model.add(layers.Dense(32,kernel_initializer='random_normal',kernel_regularizer=tf.keras.regularizers.l2(0.03)))
model.add(layers.Dense(1,kernel_initializer='random_normal',kernel_regularizer=tf.keras.regularizers.l2(0.03)))
tf.keras.backend.set_floatx('float64')
# compile相当于对网络进行配置，指定好优化器和损失函数等
model.compile(optimizer=tf.keras.optimizers.SGD(0.001),loss='mean_squared_error')
# 训练
# 所有的x ; y ; 0.25的验证集 ; 整个数据集遍历10遍 ; 每一次迭代64个样本
# 是当然也可以 手动划分训练集和测试集 分别传入
model.fit(input_features,labels,validation_split=0.25,epochs=100,batch_size=64)

# 展示网络结构
model.summary()

# 预测模型结果
predict = model.predict(input_features)
print(predict)

# 测试结果并进行展示
dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year,month,day in zip(years,months,days)]
dates = [datetime.datetime.strptime(date,'%Y-%m-%d') for date in dates]

# 创建一个表格来存日期和其对应的标签数值
true_data = pd.DataFrame(data={'date':dates,'actual':labels})

# 同理，再创键一个来存日期和其对应的模型预测值
months = features[:,features_list.index('month')]
days = features[:,features_list.index('day')]
years = features[:,features_list.index('year')]

test_dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year,month,day in zip(years,months,days)]

test_dates = [datetime.datetime.strptime(date,'%Y-%m-%d') for date in test_dates]

predictions_data = pd.DataFrame(data={'date':test_dates,'prediction':predict.reshape(-1)})

# 真实值
plt.plot(true_data['date'],true_data['actual'],'b-',label = 'actual')

# 预测值
plt.plot(predictions_data['date'],predictions_data['prediction'],'ro',label = 'prediction')
plt.xticks(rotation =60)
plt.legend()
# 图名
plt.xlabel('Date')
plt.ylabel("Maximum Temperature(F)")
plt.title('Actual and Predicted Values')
plt.show()