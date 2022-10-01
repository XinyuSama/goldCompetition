"""
 * description: fashion模型导入进行测试
 * date: 2022/10/02/00:57:00
 * author: xin yu
 * version: 1.0
"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import fashion as fa

model = keras.models.load_model('fashion_model.h5')

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# 测试的时候需要对测试数据进行和训练的时候相同的预处理
# 预处理
train_images = train_images / 255.0
test_images = test_images / 255.0

# 预测结果
predictions = model.predict(test_images)
print(predictions.shape)

num_rows = 5
num_cols = 3
num_images = num_cols * num_rows
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    fa.plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    fa.plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

