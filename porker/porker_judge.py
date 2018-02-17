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
from util_trainer import trainer
from sklearn.cross_validation import train_test_split
from common.multi_layer_net_extend import MultiLayerNetExtend

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
x_train, x_test, t_train, t_test = train_test_split(x_data, t_data, test_size=0.2, random_state=1234)
# MNIST
# (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

hidden_size_list = [30, 50, 100]
NN = neural_network(input_size=10, hidden_size_list=hidden_size_list, output_size=10,
                    weight_init_std=0.01, use_batchnorm=True, use_dropout=False, dropout_ration=0.5)


better_acc = 0
for i in range(50):
    hidden_layer_num = np.random.randint(low=1, high=10, size=1)
    hidden_neuron_num =  np.random.randint(low=1, high=100, size=1)
    hidden_size_list = []
    for k in range(hidden_layer_num[0]):
        hidden_size_list.append(hidden_neuron_num[0])
    #optimizer = SGD(lr=lr)
    # optimizer = Momentum()
    optimizer = Adam()
    T = trainer(training_data=x_train, teacher_data=t_train, max_epochs=50, batch_size=100)
    NN = neural_network(input_size=10, hidden_size_list=hidden_size_list, output_size=10,
                    weight_init_std=0.01, use_batchnorm=True, use_dropout=False, dropout_ration=0.5)
    acc = T.training(neural_network=NN, optimizer=optimizer)
    print("accuracy : " + str(acc) + " | layer : " + str(hidden_layer_num[0]) + ", neuron : " + str(hidden_neuron_num[0]))
    if(better_acc < acc):
        better_acc = acc
        T.store_model("/model/porker")



