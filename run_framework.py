import data_loader_nordic_runes as dat
import nn_framework.framework as framework
import nn_framework.layer as layer
import nn_framework.activation as activation
import nn_framework.error_fun as error_fun
import autoencoder_viz as viz
from nn_framework.regularization import L1, L2, Limit
import time
import math


starting_time = time.time()


input_pixel_range = [0, 1]  # range of values of input pixels
normalized_pixel_range = [-.5, .5]  # tuning range for where to normalize input
n_hidden_nodes = [24]

training_set, evaluation_set = dat.get_data_sets()

sample = next(training_set())
n_pixels = sample.shape[0] * sample.shape[1]
printer = viz.Printer(input_shape=sample.shape)

n_nodes = [n_pixels] + n_hidden_nodes + [n_pixels]
dropout_rates = [.2, .5]

model = []
for i_layer in range(len(n_nodes)-1):
    new_layer = layer.Dense(
        n_nodes[i_layer],
        n_nodes[i_layer + 1],
        activation.tanh,
        dropout_rate=dropout_rates[i_layer]
    )
    # new_layer.add_regularizer(L1())
    # new_layer.add_regularizer(L2())
    new_layer.add_regularizer(Limit())
    model.append(new_layer)


autoencoder = framework.ANN(
    model=model,
    error_function=error_fun.abs,
    printer=printer,
    normalized_pixel_range=normalized_pixel_range,
    input_pixel_range=input_pixel_range
)
autoencoder.train(training_set)
autoencoder.evaluate(evaluation_set)


ending_time = time.time()  # Time after it finished
print('total program runtime:  ', math.floor(int(ending_time - starting_time)/60),
      'm ', (int(ending_time-starting_time) % 60), 's  \n\n')  # print the time taken to run
