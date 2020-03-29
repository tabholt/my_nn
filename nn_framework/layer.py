import numpy as np


class Dense(object):
    def __init__(self, m_inputs, n_outputs, activation_function, learning_rate=.05, initial_weight_scale=1):
        self.m_inputs = int(m_inputs)
        self.n_outputs = int(n_outputs)
        self.learning_rate = learning_rate
        self.initial_weight_scale = initial_weight_scale
        self.activation_function = activation_function

        # Choose random weights
        # inputs match to m rows outputs to n columns
        # Add one row for the bias term
        self.weights = np.random.rand(
            m_inputs+1, n_outputs)*self.initial_weight_scale*2-1
        self.x = np.zeros((1, m_inputs+1))
        self.y = np.zeros((1, n_outputs))

    def forward_prop(self, in_array):
        bias = np.ones((1, 1))
        self.x = np.concatenate((in_array, bias), axis=1)
        v = np.matmul(self.x, self.weights)
        self.y = self.activation_function.calc(v)
        return self.y

    def back_prop(self, de_dy):
        dy_dv = self.activation_function.calc_d(self.y)
        dv_dx = self.weights.transpose()
        de_dx = np.matmul(de_dy * dy_dv, dv_dx)
        return de_dx[:-1]  # dont change bias node
