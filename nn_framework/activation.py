import numpy as np


class tanh(object):
    @staticmethod
    def calc(v):
        return np.tanh(v)

    @staticmethod
    def calc_d(v):
        return 1 - np.tanh(v)**2


class logistic(object):
    @staticmethod
    def calc(v):
        return 1/(1 + np.exp(-v))

    @staticmethod
    def calc_d(v):
        fv = 1/(1 + np.exp(-v))
        return fv * (1-fv)


class relu(object):
    @staticmethod
    def calc(v):
        return np.maximum(0, v)

    @staticmethod
    def calc_d(v):
        d = np.zeros(v.shape)
        d[np.where(v > 0)] = 1
        return d
