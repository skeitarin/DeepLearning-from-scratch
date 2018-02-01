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

print("--- cross_entropy_error2 ---")
print(cross_entropy_error2(y1, t))

# 微分
def numerical_diff(f, x):
    h = 1e-4 # 0.0001
    return (f(x+h) - f(x-h)) / (2*h)

def f1(x):
    return 0.01 * x ** 2 + 0.1 * x # y=0.01x2 + 0.1x

def tangent_line(f, x):
    d = numerical_diff(f, x)
    print(d)
    y = f(x) - d*x
    return lambda t: d*t + y


import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0.0, 20.0, 0.1)
y = f1(x)
tf = tangent_line(f1, 5)
y2 = tf(x)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x, y)
plt.plot(x, y2)
# plt.show()

# 偏微分
def f2(x):
    return x[0]**2 + x[1]**2

def f_tmp1(x0):
    return x0**2 + 4**2

def f_tmp2(x1):
    return x1**2 + 3**2

# 偏微分
# 一つの変数を除き、定数とすることで微分を行う
tmp1 = numerical_diff(f_tmp2, 4)
tmp2 = numerical_diff(f_tmp2, 3)

print(tmp1)
print(tmp2)
print(tmp1 + tmp2) # 偏微分＋偏微分＝全微分（接平面の傾き）

# 勾配（Gradient）
print("--- 勾配法（Gradient） ---")
def numerical_gradient(f, x):
    # h = 1e-4 # 0.0001
    # grad = np.zeros_like(x) # xと同じ型の配列を生成

    # for i in range(x.size):
    #     print("x:" + str(x) + ", i:" + str(i))
    #     tmp = x[i]
    #     # f(x+h)の計算
    #     x[i] = tmp + h
    #     f_1 = f(x)
        
    #     # f(x-h)の計算
    #     x[i] = tmp - h
    #     f_2 = f(x)

    #     grad[i] = (f_1 - f_2) / (2 * h) # 偏微分（x[i]に限りなく0に近い値を増減し傾きを計算。その他は定数化）
    #     x[i] = tmp
    # return grad
    
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 値を元に戻す
        it.iternext()   
        
    return grad

print(numerical_gradient(f2, np.array([3.0, 4.0]))) 

# 勾配降下法（Gradient Descent）
print("--- 勾配法（Gradient Descent） ---")
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr*grad

    return x

def f3(x):
    return x[0]**2 + x[1]**2

print(gradient_descent(f3, np.array([-3.0, 4.0]), lr=0.1, step_num=100))


# ニューラルネットワークの学習
# 2×3のネットワークを想定
print("--- ニューラルネットワークの学習 ---")
class simpleNN:
    def __init__(self):
        self.W = np.random.randn(2, 3) # ガウス分布で重みを初期化
    
    def predict(self, x):
        return np.dot(x, self.W)

    def softmax_function(self, a):
        c = np.max(a)
        e_a = np.exp(a - c) # オーバーフロー対策
        sum_e_a = np.sum(e_a)
        return e_a / sum_e_a

    def loss(self, x, t):
        z = self.predict(x)
        y = self.softmax_function(z)
        loss = cross_entropy_error2(y, t)
        return loss

NN = simpleNN()
print("weight : " + str(NN.W))
x = np.array([0.6, 0.9]) # 予測値
print(x.shape)
p = NN.predict(x)
print("predict : " + str(p))
print(np.argmax(p))
t = np.array([0, 0, 1]) # 正解ラベル
print("loss : " + str(NN.loss(x, t)))

def f(W):
    return NN.loss(x, t)

dW = numerical_gradient(f, NN.W)
print(" ------------ ")
print(dW)
# print("gradient : " + str(dW))