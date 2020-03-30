import data_loader_nordic_runes as dat
import nn_framework.framework as framework
import nn_framework.layer as layer
import nn_framework.activation as activation
import nn_framework.error_fun as error_fun
import autoencoder_viz as viz

input_pixel_range = [0, 1]  # range of values of input pixels
normalized_pixel_range = [-.5, .5]  # tuning range for where to normalize input
n_hidden_nodes = [24]

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

printer = viz.Printer(input_shape=sample.shape)
try:
    os.mkdir('nn_images')
except:
    pass

autoencoder = framework.ANN(
    model=model,
    error_function=error_fun.sqr,
    printer=printer,
    normalized_pixel_range=normalized_pixel_range,
    input_pixel_range=input_pixel_range
)
autoencoder.train(training_set)
autoencoder.evaluate(evaluation_set)
