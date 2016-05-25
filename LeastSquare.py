import numpy as np
import sys
import math
class LeastSquare:

    def __init__(self, X_, Y_):
        self.X = X_
        self.Y = Y_
        self.training = [self.X[0:100], self.Y[0:100]]
        self.validation = [self.X[100:200], self.Y[100:200]]
        self.test = [self.X[200:300], self.Y[200:300]]


    def psi(self, x_, n):
        arr = []
        for i in range(n):
            arr.append(math.pow(x_,i))
        return arr

    def trasnform_set(self, points):
        arr = []
        for i in range(len(points[0])):
            arr.append([self.psi(points[0][i], 15), points[1][i]])
        return arr


if __name__ == "__main__":
    X = np.load(sys.argv[1])
    Y = np.load(sys.argv[2])
    ls = LeastSquare(X,Y)

    print (ls.trasnform_set(ls.training))


