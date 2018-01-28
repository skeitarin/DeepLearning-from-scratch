import sys, os
sys.path.append("./deep-learning-from-scratch-master/")
import numpy as np
import pickle
from dataset.mnist import load_mnist
from PIL import Image

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

img = x_train[0]
lbl = t_train[0]

print(img.shape)
img = img.reshape(28, 28) # 形状を元の画像サイズ（28×28）に戻す
print(img.shape)

# img_show(img)

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True, one_hot_label=False)
    return x_test, t_test

def init_nn():
    with open("./deep-learning-from-scratch-master/ch03/sample_weight.pkl", "rb") as f:
        nn = pickle.load(f)
    return nn

def sigmoid_function(x):
    return 1 / (1 + np.exp(-x)) 

def softmax_function(a):
    c = np.max(a)
    e_a = np.exp(a - c) # オーバーフロー対策
    sum_e_a = np.sum(e_a)
    return e_a / sum_e_a

def predict(nn, x):
    W1, W2, W3 = nn["W1"], nn["W2"], nn["W3"]
    b1, b2, b3 = nn["b1"], nn["b2"], nn["b3"]

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid_function(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid_function(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax_function(a3)
    return y

x, t = get_data()
nn = init_nn()
batxh_size = 100 # 一度にまとめて予測する数（バッチ数）
accuracy_cnt = 0
# for i in range(len(x)):
#     y = predict(nn, x[i])
#     p = np.argmax(y) # 最も確率が高い要素のインジェクションを取得
#     if p == t[i]:
#         accuracy_cnt += 1
for i in range(0, len(x), batxh_size): # range(start, end, step) step:増加数
    x_batch = x[i:i + batxh_size]
    # 一度にまとめて予測処理を行うほうが効率が良い
    y_batch = predict(nn, x_batch)
    p = np.argmax(y_batch, axis=1) # 比較する次元の単位を指定（この場合、１次元ごとに最大値をもつインデックスを返す）
    accuracy_cnt += np.sum(p == t[i:i + batxh_size])

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))

x = np.array([[0.1, 0.8, 0.1], [0.1, 0.1, 0.6], [0.6, 0.1, 0.1]])
print(np.argmax(x)) # 1(値:0.8） <- 次元関係なく、最大値
print(np.argmax(x, axis=1)) # [1, 2, 0]（値:0.8, 0.6, 0.6） <-１次元ごとに最大値をもつインデックスを返す 

