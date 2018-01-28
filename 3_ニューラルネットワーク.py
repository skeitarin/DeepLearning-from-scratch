import numpy as np
import matplotlib.pyplot as plt

# 活性化関数

def plot_graph(x, y):
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)
    plt.show()

# -------------
# ステップ関数
# -------------
print("--- step function ---")
def step_function(x):
    if(x > 0):
        return 1
    else:
        return 0
    # ↑引数のxは実数のみ、ndarrayなどは渡せない

x1 = 0.1
x2 = np.array([1, -1])

print(step_function(x1))
# print(step_function(x2)) #-> Error

def step_function2(x):
    x2 = x > 0
    return x2.astype(np.int)

print(step_function2(x2))

x3 = np.arange(-5.0, 5.0, 0.1)
y = step_function2(x3)

# plot_graph(x3, y)


# -------------
# シグモイド関数
# -------------
print("--- sigmoid function ---")
def sigmoid_function(x):
    return 1 / (1 + np.exp(-x)) # exp(x)はネイピア数（e）のx乗

print(sigmoid_function(np.arange(-0.1, 1.0, 1)))
y = sigmoid_function(x3)

# plot_graph(x3, y)

# -------------
# ReLU関数
# -------------
print("--- ReLU function ---")
def relu_function(x):
    return np.maximum(0, x)

y = relu_function(x3)
# plot_graph(x3, y)

# -------------
# 多次元配列
# -------------
print("--- 多次元配列 ---")
A = np.array([[1,2], [3,4]])
B = np.array([[5,6], [7,8]])
print(A.shape)
print(B.shape)
# 行列の積（多次元配列のドット積）
print(np.dot(A, B))

C = np.array([[1,2,3], [4,5,6]])
D = np.array([[1,2], [3,4], [5,6]])
print(C.shape)
print(D.shape)
print(np.dot(C, D))

# -------------
# ニューラルネットワーク
# -------------
print("--- neural network ---")
x = np.array([1, 2])
w = np.array([[1,3,5], [2,4,6]])
print(x.shape)
print(w.shape)
print(np.dot(x, w))

# 入力層 -> 第１層
print("入力層 -> 第１層")
x = np.array([1.0, 0.5]) # input
w1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]) # weight
b1 = np.array([0.1, 0.2, 0.3]) # bias

a1 = np.dot(x, w1) + b1
print(a1)
z1 = sigmoid_function(a1)
print(z1)

# 第１層 -> 第２層
print("第１層 -> 第２層")
w2 = np.array([[0.1, 0.3], [0.2, 0.5], [0.3, 0.6]]) # ニュートン数が第１層が３、第２層が２のため
b2 = np.array([0.1, 0.2])

a2 = np.dot(z1, w2) + b2
print(a2)
z2 = sigmoid_function(a2)
print(z2)
# 第２層 -> 出力層
print("第２層 -> 出力層")
w3 = np.array([[0.1, 0.3], [0.2, 0.4]])
b3 = np.array([0.1, 0.2])

a3 = np.dot(z2, w3) + b3
print(a3)

def identity_function(x):
    return x

y = identity_function(a3)
print(y)

# まとめ
print("まとめ")

class neural_network:
    def __init__(self):
        self.W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
        self.W2 = np.array([[0.1, 0.3], [0.2, 0.5], [0.3, 0.6]])
        self.W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
        self.b1 = np.array([0.1, 0.2, 0.3])
        self.b2 = np.array([0.1, 0.2])
        self.b3 = np.array([0.1, 0.2])
    
    def forward(self, x):
        a1 = np.dot(x, self.W1) + self.b1
        z1 = sigmoid_function(a1)
        a2 = np.dot(z1, self.W2) + self.b2
        z2 = sigmoid_function(a2)
        a3 = np.dot(z2, self.W3) + self.b3
        y = identity_function(a3)
        return y

NN = neural_network()
x = np.array([1.0, 0.5])
y = NN.forward(x)
print(y)

# -------------
# softmax関数
# -------------
print("softmax関数")
def softmax_function(a):
    c = np.max(a)
    e_a = np.exp(a - c) # オーバーフロー対策
    sum_e_a = np.sum(e_a)
    return e_a / sum_e_a

a = np.array([0.3, 2.9, 4.0])
y = softmax_function(a)
print(y)
print(np.sum(y))