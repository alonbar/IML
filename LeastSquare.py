import numpy as np
import sys
import math
from matplotlib import pyplot

class LeastSquare:
    def __init__(self, X_, Y_):
        self.X = X_
        self.Y = Y_
        self.training = [self.X[0:100], self.Y[0:100]]
        self.validation = [self.X[100:200], self.Y[100:200]]
        self.test = [self.X[200:300], self.Y[200:300]]


    def psi(self, x_, n_):
        arr = []
        for i in range(n_):
            arr.append(math.pow(x_,i))
        return arr

    def trasnform_set(self, points_):
        arr = []
        for i in range(len(points_[0])):
            arr.append([self.psi(points_[0][i], 16), points_[1][i]])
        return arr

    def initialize_matrices(self, initialized_training_set_):
        x_matrix = []
        y_vector = []
        for item in initialized_training_set_:
            x_matrix.append(item[0])
            y_vector.append(item[1])
        x_matrix = np.transpose(x_matrix)
        return x_matrix, y_vector

    def find_hypothesis_series(self, x_matrix_, y_vector_, dimension_):
        hypothesis_series = []
        x_tpi = np.transpose(np.linalg.pinv(x_matrix_))
        for i in range(dimension_):
            hypothesis_series.append(np.dot(np.transpose(np.linalg.pinv(x_matrix_[:i+1])), y_vector_))
        return  hypothesis_series

    def find_error(self,w_, x_set_, y_set_):
        w_on_x = np.dot(np.transpose(x_set_), w_)
        temp_result = np.subtract(w_on_x, y_set_)
        return math.pow(np.linalg.norm(temp_result), 2)

    def find_best_hypothesis(self, hypothesis_series, x_validation_, y_validation_ ):
        loss_array = []
        for i,w in enumerate(hypothesis_series):
            loss_array.append(self.find_error(w,x_validation_[:len(w)], y_validation_))
        return hypothesis_series[np.argmin(loss_array)], loss_array

    def test_error(self, h, x_test, y_test):
        return self.find_error(h, x_test[:len(h)], y_test)/len(y_test)

    def k_fold(self,k, dimension_):
        initialized_set = self.trasnform_set([self.X[:200], self.Y[:200]])
        #splitting to k groups

        j = 0
        split_set = []
        split_set.append( [])
        for i in range(len(initialized_set)):
            split_set[j].append(initialized_set[i])
            if i != 0 and (i % int(len(initialized_set)/k)  == 0):
                j = j + 1
                split_set.append([])
        error_list = []
        for i in range(5):
            current_A_input = []
            for j in range(5):
                if j == i:
                    continue
                current_A_input += split_set[i]
            x_matrix_train, y_vector_train = self.initialize_matrices(current_A_input)
            hypothsis_list = self.find_hypothesis_series(x_matrix_train, y_vector_train, dimension_)
            x_matrix_validation, y_vector_validation = self.initialize_matrices(split_set[i])
            minimum_error_h= self.find_best_hypothesis(hypothsis_list,  x_matrix_validation, y_vector_validation)[0]
            h_error = self.test_error(minimum_error_h, x_matrix_validation, y_vector_validation)
            error_list.append(h_error)

        print(error_list[np.argmin(error_list)])

    def manager(self, dimension_):
        #training - getting different hypothsis for differnt dimensiosn
        initialized_training_set = self.trasnform_set(self.training)
        x_matrix_training_set, y_vector_training_set = self.initialize_matrices(initialized_training_set)
        hypothesis_series = self.find_hypothesis_series(x_matrix_training_set, y_vector_training_set, dimension_)
        #finind h* using the validation set
        initialized_validation_set = self.trasnform_set(self.validation)
        x_validation, y_validation = self.initialize_matrices(initialized_validation_set)
        minimum_error_h, loss_array= self.find_best_hypothesis(hypothesis_series, x_validation, y_validation)
        #testing h* - finding it's error

        initialize_test_set = self.trasnform_set(self.test)
        x_test_set, y_test_vector = self.initialize_matrices(initialize_test_set)
        h_error = self.test_error(minimum_error_h, x_test_set, y_test_vector)
        self.k_fold(5, dimension_)
        pyplot.plot([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],loss_array[1:],'kv')
        pyplot.show()
        pyplot.savefig


if __name__ == "__main__":
    X = np.load(sys.argv[1])
    Y = np.load(sys.argv[2])
    ls = LeastSquare(X,Y)
    ls.manager(16)
    br = ls.trasnform_set(ls.training)



