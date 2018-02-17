import os,sys
from os import path
import pickle
import numpy as np

class model_manager():
    def __init__(self, file_path):
        extention = ".pickle"
        current_dir = path.dirname(path.abspath( __file__ ))
        self.pickle_file = current_dir + file_path + extention

    def store(self, model, overwrite=True):
        if(os.path.exists(self.pickle_file) == True):
            if(overwrite == True):
                os.remove(self.pickle_file)
            else:
                return
        with open(self.pickle_file, 'wb') as f:
            pickle.dump(model, f)
    
    def load(self):
        with open(self.pickle_file, 'rb') as f:
            tmp = pickle.load(f)
        return tmp
