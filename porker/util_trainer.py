import sys,os
import numpy as np
from util_model_manager import model_manager

class trainer():
    def __init__(self, training_data, teacher_data, max_epochs, batch_size, display_accuracy_change=False):
        self.training_data = training_data
        self.teacher_data = teacher_data
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.display_accuracy_change = display_accuracy_change

    def training(self, neural_network, optimizer):
        self.neural_network = neural_network
        self.accuracy_list = []
        epoch_cnt = 1

        train_size = self.training_data.shape[0]
        iter_per_epoch = max(self.batch_size*100 / self.batch_size, 1)
        for i in range(10000000):
            batch_mask = np.random.choice(train_size, self.batch_size)
            x_batch = self.training_data[batch_mask]
            t_batch = self.teacher_data[batch_mask]

            # 誤差逆伝播法による勾配の計算
            grads = neural_network.gradient(x_batch, t_batch)
            # 重みの調整
            optimizer.update(neural_network.params, grads)
            if i % iter_per_epoch == 0:
                acc = neural_network.accuracy(self.training_data, self.teacher_data)
                self.accuracy_list.append(acc)
                if(self.display_accuracy_change):
                    print("epoch : " + str(epoch_cnt) + " accuracy : " + str(acc))

                epoch_cnt += 1
                if epoch_cnt >= self.max_epochs:
                    break
        
        return neural_network.accuracy(self.training_data, self.teacher_data)

    def store_model(self, output_model_path):
        # 学習したニューラルネットワークを保存
        manager = model_manager(output_model_path)
        manager.store(self.neural_network, overwrite=True)

    def display_graph(self):
        x = np.arange(len(self.accuracy_list))
        plt.plot(x, self.accuracy_list, label='train acc')
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.ylim(0, 1.0)
        plt.legend(loc='lower right')
        plt.show()
        