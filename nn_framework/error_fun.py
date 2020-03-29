import numpy as np


class sqr(object):
    @staticmethod
    def calc(x, y):
        return (y-x)**2

    @staticmethod
    def calc_d(x, y):
        return 2 * (y-x)


class abs(object):
    @staticmethod
    def calc(x, y):
        return np.abs(y-x)

    @staticmethod
    def calc_d(x, y):
        return np.sign(y-x)
