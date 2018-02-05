import sys, os
sys.path.append("./deep-learning-from-scratch-master/")
import numpy as np
import pickle
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist

class TwoLayerNN:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 重みの初期化
        self.params = {}
        self.params["W1"] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b2"] = np.zeros(output_size)
    
    def sigmoid_function(self, x):
        return 1 / (1 + np.exp(-x)) # exp(x)はネイピア数（e）のx乗

    def softmax_function(self, a):
        c = np.max(a)
        e_a = np.exp(a - c) # オーバーフロー対策
        sum_e_a = np.sum(e_a)
        return e_a / sum_e_a

    def cross_entropy_error(self, y, t):
        if y.ndim == 1:
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)
        # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
        if t.size == y.size:
            t = t.argmax(axis=1) 
        batch_size = y.shape[0]
        return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

    def predict(self, x):
        W1, W2 = self.params["W1"], self.params["W2"]
        b1, b2 = self.params["b1"], self.params["b2"]

        a1 = np.dot(x, W1) + b1
        z1 = self.sigmoid_function(a1)
        a2 = np.dot(z1, W2) + b2
        y = self.softmax_function(a2)
        return y

    def loss(self, x, t):
        y = self.predict(x)
        return self.cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, f, x):
        h = 1e-4 # 0.0001
        grad = np.zeros_like(x)
        it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            tmp_val = x[idx]
            x[idx] = float(tmp_val) + h # self.params["x"]の更新（numpyのため参照渡しになる）
            fxh1 = f(x) # f(x+h) -> xはダミーの引数。loss引数xには、gradient引数xが渡される
        
            x[idx] = tmp_val - h 
            fxh2 = f(x) # f(x-h)
            grad[idx] = (fxh1 - fxh2) / (2*h)
        
            x[idx] = tmp_val # 値を元に戻す
            it.iternext()   
        return grad

    def gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        grads = {}
        grads["W1"] = self.numerical_gradient(loss_W, self.params["W1"])
        grads["b1"] = self.numerical_gradient(loss_W, self.params["b1"])
        grads["W2"] = self.numerical_gradient(loss_W, self.params["W2"])
        grads["b2"] = self.numerical_gradient(loss_W, self.params["b2"])
        return grads

# NN = TwoLayerNN(input_size=784, hidden_size=100, output_size=10)
# x = np.random.rand(100, 784)
# t = np.random.rand(100, 10)
# y = NN.predict(x)
# grads = NN.gradient(x, t)
# print(grads["W1"].shape)

# --- ミニバッチ学習 ---
print("--- mini batch learning ---")
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, one_hot_label=True)

train_loss_list = []
train_acc_list = []
test_acc_list = []

# ハイパーパラメータ
iters_num = 10
train_size = x_train.shape[0]
batch_size = 10
learning_rate = 0.1

# 1エポックあたりの繰り返し数
iter_per_epoch = max(train_size / batch_size, 1)

NN = TwoLayerNN(input_size=784, hidden_size=50, output_size=10)

for i in range(iters_num):
    # ミニバッチの取得
    batch_mask = np.random.choice(train_size, batch_size)
    # print("batch_mask : " + str(batch_mask))
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 勾配の計算
    grads = NN.gradient(x_batch, t_batch)

    # パラメータの更新
    for key in ("W1", "W2", "b1", "b2"):
        NN.params[key] -= learning_rate * grads[key]
    
    # 学習経過の記録
    loss = NN.loss(x_batch, t_batch)
    print("- learning... " + str(i) + "/" + str(iters_num) + " - loss:" + str(loss))
    train_loss_list.append(loss)

    # 1エポックごとに認識精度を計算
    if 1 % iter_per_epoch == 0:
        train_acc_list.append(NN.accuracy(x_train, t_train))
        test_acc_list.append(NN.accuracy(x_train, t_train))
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))


# グラフの描画
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()