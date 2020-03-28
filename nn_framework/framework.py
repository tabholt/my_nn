import numpy as np 

class ANN(object):
    def __init__(self, model=None):
        self.layers = model
        self.n_iter_train = int(1e3)
        self.n_iter_evaluate = int(1e3)
    
    def train(self, training_set=None):
        for i in range(self.n_iter_train):
            x = next(training_set()).ravel()
            print(x)

    def evaluate(self, evaluation_set=None):
        for i in range(self.n_iter_evaluate):
            x = next(evaluation_set()).ravel()
            print(x)