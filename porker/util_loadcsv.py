import os,sys
from os import path
import pickle
import numpy as np

class load_as_numpy():
    def __init__(self):
        self.current_dir = path.dirname(path.abspath( __file__ ))
    def load(self, file_path):
        pickle_file = self.current_dir + file_path + ".pickle"
        # 初回のみpickleファイルを作成->２回目移行のload処理を高速化
        if(os.path.exists(pickle_file) == False):
            with open(pickle_file, "wb") as f:
                pickle.dump(np.loadtxt(self.current_dir + file_path, delimiter=","), f)
        with open(pickle_file, 'rb') as f:
            tmp = pickle.load(f)
        return tmp
