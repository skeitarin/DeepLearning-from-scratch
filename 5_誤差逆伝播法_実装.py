import sys, os
sys.path.append("./deep-learning-from-scratch-master/")
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from datetime import datetime

# ReLUレイヤー
class ReluLayer:
    def __init__(self):
        self.mask = None # numpyのBoolean
    def forward(self, x):
        self.mask = (x <= 0) # x<=0の箇所をTrue、それ以外をFalseのnumpy配列（次元は引数xと同じ）
        out = x.copy()
        out[self.mask] = 0 # Trueの箇所（0以下）を0に変更する、その他はそのまま（ReLu関数）
        return out
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx

# Affineレイヤー
class AffineLayer():
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None
    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx

# Softmax-with-Lossレイヤー（Lossは交差エントロピー誤差）
class SoftmaxWithLossLayer():
    def __init__(self):
        self.loss = None # 損失
        self.y = None # Softmax関数の出力
        self.t = None # 教師データ、正解ラベル（one-hot vector）
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
    def forward(self, x, t):
        self.t = t
        self.y = self.softmax_function(x)
        self.loss = self.cross_entropy_error(self.y, self.t)
        return self.loss
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx

class TwoLayerNN():
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.layers = OrderedDict()
        # 重みの初期化
        self.params["W1"] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b2"] = np.zeros(output_size)
        # レイヤの生成
        self.layers["Affine1"] = AffineLayer(self.params["W1"], self.params["b1"])
        self.layers["Relu1"] = ReluLayer()
        self.layers["Affine2"] = AffineLayer(self.params["W2"], self.params["b2"])
        self.lastLayer = SoftmaxWithLossLayer()
    
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def gradient(self, x, t):
        # forward
        self.loss(x, t)
        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        
        # 設定 
        grads = {}
        grads["W1"] = self.layers["Affine1"].dW
        grads["b1"] = self.layers["Affine1"].db
        grads["W2"] = self.layers["Affine2"].dW
        grads["b2"] = self.layers["Affine2"].db
        return grads

print("--- mini batch learning ---")
print(" proc start : " + str(datetime.now().strftime("%Y/%m/%d %H:%M:%S")))
# データの読み込み
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

NN = TwoLayerNN(input_size=784, hidden_size=50, output_size=10)

iters_num = 10000  # 繰り返しの回数を適宜設定する
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    # 勾配の計算
    #grad = NN.numerical_gradient(x_batch, t_batch)
    grad = NN.gradient(x_batch, t_batch)
    
    # パラメータの更新
    for key in ('W1', 'b1', 'W2', 'b2'):
        NN.params[key] -= learning_rate * grad[key]
    
    loss = NN.loss(x_batch, t_batch)
    print("- learning... " + str(i) + "/" + str(iters_num) + " - loss:" + str(loss))
    train_loss_list.append(loss)
    
    if i % iter_per_epoch == 0:
        train_acc = NN.accuracy(x_train, t_train)
        test_acc = NN.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

print(" proc end : " + str(datetime.now().strftime("%Y/%m/%d %H:%M:%S")))

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