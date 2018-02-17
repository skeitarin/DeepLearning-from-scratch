import os,sys
sys.path.append("./porker/") # jupyter notebook(dummy)用
sys.path.append("./deep-learning-from-scratch-master/")
from os import path
import pickle
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from util_loadcsv import load_as_numpy
from util_multi_neural_network import neural_network
from util_optimizer import *
from util_model_manager import model_manager
from sklearn.cross_validation import train_test_split
from common.multi_layer_net_extend import MultiLayerNetExtend

loader = load_as_numpy()
data = loader.load("/data/train.csv")

# 入力値 データレイアウト
# [1枚目の柄, 1枚目の数　・・・　5枚目の柄, 5枚目の数]
#  柄→1：スペード、2：クローバー、3：ダイヤ、4：ハート
x_data = data[:,0:10]  
# 入力値 データレイアウト
# [役]
#  役→0：約無し、　・・・　9：ロイヤルストレートフラッシュ
t_data = data[:,10:11].astype(np.int)
t_data = np.eye(10)[t_data.flatten()] # one-hotに変換
# 訓練データと検証データに分割
x_train, x_test, t_train, t_test = train_test_split(x_data, t_data, test_size=0.2, random_state=1234)
# MNIST
# (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

hidden_size_list = [30, 50, 100]
NN = neural_network(input_size=10, hidden_size_list=hidden_size_list, output_size=10,
                    weight_init_std=0.01, use_batchnorm=True, use_dropout=False, dropout_ration=0.5)


epoch_cnt = 1
max_epochs = 21
batch_size = 100
train_size = x_train.shape[0]
lr = 0.01
iter_per_epoch = max(batch_size*100 / batch_size, 1)
acc_list = []

#optimizer = SGD(lr=lr)
# optimizer = Momentum()
optimizer = Adam()

for i in range(1000000):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 誤差逆伝播法による勾配の計算
    grads = NN.gradient(x_batch, t_batch)
    # 重みの調整
    optimizer.update(NN.params, grads)
    if i % iter_per_epoch == 0:
        acc = NN.accuracy(x_train, t_train)
        acc_list.append(acc)
        print("epoch : " + str(epoch_cnt) + " accuracy : " + str(acc))
        
        epoch_cnt += 1
        if epoch_cnt >= max_epochs:
            break

# 学習したニューラルネットワークを保存
manager = model_manager("/model/porker")
manager.store(NN, overwrite=True)

x = np.arange(len(acc_list))
plt.plot(x, acc_list, label='train acc')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
