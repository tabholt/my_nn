import numpy as np


class sqr(object):
    @staticmethod
    def calc(x, y):
        return (y-x)**2

    @staticmethod
    def calc_d(x, y):
        return 2 * (y-x)
