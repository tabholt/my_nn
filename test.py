import numpy as np

import random

# def lottery():
#     while True:
#         yield random.randint(1, 40)


# for random_number in range(3):
#     i = next(lottery())
#     print("And the next number is... %d!" %(i))

############################################################

weights = np.random.rand(4, 3)*2-1

print(weights)

my_array = np.array([-.5, .5, .5, -.5])

my_array = np.append(my_array, 1)

print(my_array)
