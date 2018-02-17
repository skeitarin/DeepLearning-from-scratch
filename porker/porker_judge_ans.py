import os,sys
from os import path
import pickle
import numpy as np
from util_loadcsv import load_as_numpy
from sklearn.cross_validation import train_test_split
# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
sys.path.append("./deep-learning-from-scratch-master/")
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.optimizer import SGD, Adam


loader = load_as_numpy()
data = loader.load("/data/test.csv")

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
# x_train, x_test, t_train, t_test = train_test_split(x_data, t_data, test_size=0.2, random_state=1234)
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

max_epochs = 20
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.01



def __train(weight_init_std):
    network = MultiLayerNetExtend(input_size=784, hidden_size_list=[100, 100], output_size=10, 
                                    weight_init_std=weight_init_std, use_batchnorm=True)
    optimizer = SGD(lr=learning_rate)
    
    train_acc_list = []
    bn_train_acc_list = []
    
    iter_per_epoch = max(batch_size*100 / batch_size, 1)
    epoch_cnt = 0
    
    for i in range(10000):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
    
        grads = network.gradient(x_batch, t_batch)
        optimizer.update(network.params, grads)

        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            train_acc_list.append(train_acc)
            print("epoch:" + str(epoch_cnt) + " | " + str(train_acc))
    
            epoch_cnt += 1
            if epoch_cnt >= max_epochs:
                break
                
    return train_acc_list


# 3.グラフの描画==========
# weight_scale_list = np.logspace(0, -4, num=16)
__train(0.01)

# x = np.arange(max_epochs)

# for i, w in enumerate(weight_scale_list):
#     print( "============== " + str(i+1) + "/16" + " ==============")
#     train_acc_list, bn_train_acc_list = __train(w)
    
#     plt.subplot(4,4,i+1)
#     plt.title("W:" + str(w))
#     if i == 15:
#         plt.plot(x, bn_train_acc_list, label='Batch Normalization', markevery=2)
#         plt.plot(x, train_acc_list, linestyle = "--", label='Normal(without BatchNorm)', markevery=2)
#     else:
#         plt.plot(x, bn_train_acc_list, markevery=2)
#         plt.plot(x, train_acc_list, linestyle="--", markevery=2)

#     plt.ylim(0, 1.0)
#     if i % 4:
#         plt.yticks([])
#     else:
#         plt.ylabel("accuracy")
#     if i < 12:
#         plt.xticks([])
#     else:
#         plt.xlabel("epochs")
#     plt.legend(loc='lower right')
    
# plt.show()