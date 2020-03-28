import numpy as np


class ANN(object):
    def __init__(self, model=None, pixel_range=[-.5, .5]):
        self.layers = model
        self.n_iter_train = int(1e1)
        self.n_iter_evaluate = int(1e1)
        self.pixel_range = pixel_range

    def train(self, training_set=None):
        for i in range(self.n_iter_train):
            x = next(training_set()).ravel()
            x = self.normalize(x)
            print(x)

    def evaluate(self, evaluation_set=None):
        for i in range(self.n_iter_evaluate):
            x = next(evaluation_set()).ravel()
            x = self.normalize(x)
            print(x)

    def normalize(self, pic):
        """
        Transform the input picture so that all pixel values fall 
        between the desired pixel_range
        """
        high = np.max(pic)
        low = np.min(pic)
        dist = high-low

        # return a vector with each value as a percentage between max and min
        pic_percent = (pic - low) / dist
        # transform the percentage vector to desired pixel range
        pic_normalized = pic_percent * \
            (self.pixel_range[1]-self.pixel_range[0]) + self.pixel_range[0]
        return pic_normalized
