import numpy as np
import matplotlib.pyplot as plt


def get_data_sets():
    """
    This function creates two other functions that generate data.
    One generates a training data set and the other, an evaluation set.

    The examples have the format of a two-dimensional numpy array.
    They can be thought of as a very small (two-pixel by two-pixel) image.


    To use in a script:

        import data_loader_two_by_two as dat

        training_generator, evaluation_grenerator = dat.get_data_sets()
        new_training_example = training_generator.next()
        new_evaluation_example = evaluation_generator.next()
    """

    examples = [
        np.array([
            [1, 1, 1],
            [0, 0, 0],
            [0, 0, 0],
        ]),
        np.array([
            [0, 0, 0],
            [1, 1, 1],
            [0, 0, 0],
        ]),
        np.array([
            [0, 0, 0],
            [0, 0, 0],
            [1, 1, 1],
        ]),
        np.array([
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
        ]),
        np.array([
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
        ]),
        np.array([
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
        ]),
        np.array([
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
        ]),
        np.array([
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
        ]),
        np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]),
        np.array([
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
        ]),
        np.array([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 1],
        ]),
        np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
        ]),
        np.array([
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ]),
        np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ]),
        np.array([
            [1, 1, 1],
            [1, 0, 0],
            [1, 0, 0],
        ]),
        np.array([
            [0, 0, 1],
            [0, 0, 1],
            [1, 1, 1],
        ]),
        np.array([
            [1, 1, 1],
            [0, 0, 1],
            [0, 0, 1],
        ]),
        np.array([
            [1, 0, 0],
            [1, 0, 0],
            [1, 1, 1],
        ]),
    ]

    def training_set():
        while True:
            index = np.random.choice(len(examples))
            yield examples[index]

    def evaluation_set():
        while True:
            index = np.random.choice(len(examples))
            yield examples[index]

    # def visualize_set():
    #     data = np.random.permutation(examples)
    #     for example in data:
    #         plt.imshow(example)
    #         plt.show()

    return training_set, evaluation_set


# train, evalu, vizu = get_data_sets()

# vizu()
