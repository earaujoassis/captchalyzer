import numpy as np


rng = np.random.RandomState(3141592653)


class DataFile(object):
    def __init__(self, X, y=None, expectation=None):
        self.X = X
        self.y = y
        self.expectation = expectation
