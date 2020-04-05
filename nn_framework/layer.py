import numpy as np


class Dense(object):
    def __init__(
        self,
        m_inputs,
        n_outputs,
        activation_function,
        learning_rate=.001,
        initial_weight_scale=1,
        dropout_rate=0
    ):
        self.m_inputs = int(m_inputs)
        self.n_outputs = int(n_outputs)
        self.learning_rate = learning_rate
        self.initial_weight_scale = initial_weight_scale
        self.activation_function = activation_function
        self.dropout_rate = dropout_rate

        # Choose random weights
        # inputs match to m rows outputs to n columns
        # Add one row for the bias term
        self.weights = np.random.rand(
            m_inputs+1, n_outputs)*self.initial_weight_scale*2-1
        self.x = np.zeros((1, self.m_inputs+1))
        self.y = np.zeros((1, self.n_outputs))

        self.regularizers = []

    def add_regularizer(self, new_regularizer):
        self.regularizers.append(new_regularizer)

    def forward_prop(self, in_array, evaluating=False):
        if evaluating:
            dropout_rate = 0
        else:
            dropout_rate = self.dropout_rate

        self.i_dropout = np.zeros(self.x.size, dtype=bool)
        self.i_dropout[np.where(
            np.random.uniform(size=self.x.size) < dropout_rate)] = True
        self.x[:, self.i_dropout] = 0
        self.x[:, np.logical_not(self.i_dropout)] *= 1 / (1 - dropout_rate)

        bias = np.ones((1, 1))
        self.x = np.concatenate((in_array, bias), axis=1)
        v = np.matmul(self.x, self.weights)
        self.y = self.activation_function.calc(v)
        return self.y

    def back_prop(self, de_dy):
        # v = self.x @ self.weights
        # dv_dw = self.x
        # dv_dx = self.weights
        dy_dv = self.activation_function.calc_d(self.y)
        dv_dx = self.weights.transpose()
        dy_dw = np.matmul(self.x.transpose(), dy_dv)
        de_dw = de_dy * dy_dw
        self.weights -= de_dw * self.learning_rate
        for regularizer in self.regularizers:
            self.weights = regularizer.update(self)

        de_dx = np.matmul(de_dy * dy_dv, dv_dx)
        # remove dropped out inputs from this run
        de_dx[:, self.i_dropout] = 0
        return de_dx[:, :-1]  # dont change bias node
