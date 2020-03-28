import data_loader_two_by_two as dat
import nn_framework.framework as framework
import nn_framework.layer as layer
import nn_framework.activation as activation

input_pixel_range = [0, 1]  # range of values of input pixels
normalized_pixel_range = [-.5, .5]  # tuning range for where to normalize input
n_hidden_nodes = [7, 4, 6]

training_set, evaluation_set = dat.get_data_sets()

sample = next(training_set())
n_pixels = sample.shape[0] * sample.shape[1]
n_nodes = [n_pixels] + n_hidden_nodes + [n_pixels]

model = []
for i_layer in range(len(n_nodes)-1):
    model.append(layer.Dense(
        n_nodes[i_layer],
        n_nodes[i_layer + 1],
        activation.tanh
    ))


autoencoder = framework.ANN(
    model=model,
    normalized_pixel_range=normalized_pixel_range,
    input_pixel_range=input_pixel_range
)
autoencoder.train(training_set)
autoencoder.evaluate(evaluation_set)
