import sys, os
sys.path.append("./deep-learning-from-scratch-master/")
import numpy as np
import pickle
from dataset.mnist import load_mnist

# 二乗和誤差
def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) **2)

t = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0,]) # 正解は２
y1 = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]) # ２の確率が高い
print("--- mean_square_error ---")
print(mean_squared_error(y1, t))

y2 = np.array([0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]) # ７の確率が高い
print(mean_squared_error(y2, t))

# 交差エントロピー誤差
def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

print("--- cross_entropy_error ---")
print(cross_entropy_error(y1, t))
print(cross_entropy_error(y2, t))

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

print(x_train.shape)
print(t_train.shape)

train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size) # ランダムな値を取得
x_train_batch = x_train[batch_mask]
t_train_batch = t_train[batch_mask]

# 交差エントロピー誤差 - 改
def cross_entropy_error2(y, t):
    print(y)
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    print(y)
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size

print(cross_entropy_error2(y1, t))

