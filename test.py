import numpy as np 

import random

def lottery():
    while True:
        yield random.randint(1, 40)


for random_number in range(3):
    i = next(lottery())
    print("And the next number is... %d!" %(i))