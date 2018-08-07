import numpy as np

class DBRprop:
    def __init__(self):
        self.nbr = np.random.randint(15, 5000)
        self.n = np.random.rand(2)*5 + 0.00000001
        self.d = np.random.rand(2)*10000 + 3
        # creation of array of data representing the mirror
        self.data = np.tile(np.array([[self.n[0], self.d[0]], [self.n[1], self.d[1]]]), (self.nbr, 1))
        self.data[0, 1] = 0
        self.data[-1, 1] = 0

