import numpy as np


class ANN(object):
    def __init__(self, model, normalized_pixel_range=[-.5, .5], input_pixel_range=[np.nan, np.nan]):
        self.layers = model
        self.n_iter_train = int(1e1)
        self.n_iter_evaluate = int(1e1)
        self.normalized_pixel_range = normalized_pixel_range
        self.input_pixel_range = input_pixel_range

    def train(self, training_set):
        for i in range(self.n_iter_train):
            x = next(training_set()).ravel()
            x = self.normalize(x)
            y = self.forward_prop(x)
            print(y)

    def evaluate(self, evaluation_set):
        for i in range(self.n_iter_evaluate):
            x = next(evaluation_set()).ravel()
            x = self.normalize(x)
            y = self.forward_prop(x)
            print(y)

    def normalize(self, pic):
        """
        Transform the input picture so that all pixel values fall 
        between the desired normalized_pixel_range
        """
        if np.isnan(self.input_pixel_range[0]):
            high = np.max(pic)
            low = np.min(pic)
            dist = high-low
        else:
            high = self.input_pixel_range[1]
            low = self.input_pixel_range[0]
            dist = high-low

        # return a vector with each value as a percentage between max and min
        pic_percent = (pic - low) / dist
        # transform the percentage vector to desired pixel range
        pic_normalized = pic_percent * \
            (self.normalized_pixel_range[1]-self.normalized_pixel_range[0]
             ) + self.normalized_pixel_range[0]

        return pic_normalized

    def denormalize(self, pic, original_range=[0, 1]):
        """
        Transform the normalized pic back into its original form
        """
        high = np.max(pic)
        low = np.min(pic)
        dist = high-low

        # return a vector with each value as a percentage between max and min
        pic_percent = (pic - low) / dist
        # transform the percentage vector to desired pixel range
        pic_normalized = pic_percent * \
            (original_range[1]-original_range[0]) + original_range[0]

        return pic_normalized

    def forward_prop(self, x):
        # convert inputs into 2d array of right shape
        y = x.ravel()[np.newaxis, :]
        y = self.layers[0].forward_prop(y)
        return y.ravel()
