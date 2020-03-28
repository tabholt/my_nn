import data_loader_two_by_two as dat
import nn_framework.framework as framework

training_set, evaluation_set = dat.get_data_sets()

sample = next(training_set())
n_pixels = sample.shape[0] * sample.shape[1]
n_nodes = [n_pixels, n_pixels]

autoencoder = framework.ANN(model=None, pixel_range=[-.5, .5])
autoencoder.train(training_set)
autoencoder.evaluate(evaluation_set)

