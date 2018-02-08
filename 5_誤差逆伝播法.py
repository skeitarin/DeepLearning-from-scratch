import numpy as np

# 乗算レイヤー
class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None
    def forward(self, x, y):
        self.x = x
        self.y = y
        return x * y
    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x
        return dx, dy

# 加算レイヤー
class AddLayer:
    def __init__(self):
        pass
    def forward(self, x, y):
        return x + y
    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy

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

# Sigmoidレイヤー
class SigmoidLayer():
    def __init__(self):
        self.out = None
    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out
    def backward(self, dout):
        dx = dout * self.out * (1.0 - self.out) # p.143 ~ 146
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


apple = 100
apple_num = 2
tax = 1.1

mul_apple_lyaer = MulLayer()
mul_tax_layer = MulLayer()

# forward
apple_price = mul_apple_lyaer.forward(apple, apple_num)
price = mul_tax_layer.forward(apple_price, tax)

# backward
dprice = 1
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_num = mul_apple_lyaer.backward(dapple_price)

print(dapple, dapple_num, dtax)

# 図5-17の実装（p149）
apple_mul_layer = MulLayer()
orange_mul_layer = MulLayer()
apple_orange_add_layer = AddLayer()
tax_mul_layer = MulLayer()

apple_num = 2
apple_price = 100
orange_num = 3
orange_price = 150
tax = 1.1

# forward
apple_sum_price = apple_mul_layer.forward(apple_num, apple_price)
orange_sum_price = orange_mul_layer.forward(orange_num, orange_price)
apple_orrange_sum_price = apple_orange_add_layer.forward(apple_sum_price, orange_sum_price)
sum_price = tax_mul_layer.forward(apple_orrange_sum_price, tax)
print(sum_price)

dsum_price = 1
# backward
dapple_orrange_sum_price, dtax = tax_mul_layer.backward(dsum_price)
dapple_sum_price, dorange_sum_price = apple_orange_add_layer.backward(dapple_orrange_sum_price)
dapple_num, dapple_price = apple_mul_layer.backward(dapple_sum_price)
dorange_num, dorange_price = orange_mul_layer.backward(dorange_sum_price)
print(dapple_num, dapple_price, dorange_num, dorange_price, dtax)