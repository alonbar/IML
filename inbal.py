import numpy as np
from matplotlib import pyplot
import math
import copy
__author__ = 'Inbal'


#initializing parameters

X = np.load("C:\\Users\\baralon\\Downloads\\X_poly.npy")
Y = np.load("C:\\Users\\baralon\\Downloads\\Y_poly.npy")

def cut_set(setName, range_up, range_down):
    set_new = []
    for i in range(range_up,range_down):
        set_new.append(setName[i])
    return set_new

#this function creates vectors of powers of x for range of degrees given
def pol_reduction(set_of_x, range_of_degrees):
    reduction = [] #a set we will use to keep the new vectors - (1,x,x^2...x^n)
    current = []
    for x in set_of_x:
        for i in range(0, range_of_degrees + 1):
            current.append(math.pow(x,i))
        reduction.append(current)
        current = []
    return np.column_stack((reduction))


def least_square(x_matrix, set_of_y, degree):
    cut_x = x_matrix[0 : degree+1]
    return np.dot(np.transpose(np.linalg.pinv(cut_x)),set_of_y)


#this function validates the hypothesis received by training set
#hypothesis - array of least squares by degrees
#val_x_matrix, val_set_of_y - validation set
def validate(hypothesis, val_x_matrix, val_set_of_y):
    loss_array = calculate_error(hypothesis, val_x_matrix, val_set_of_y)
    return (np.argmin(loss_array),loss_array)

#this function performs a test on the testing set
def test(hypothesis, test_x_matrix, test_set_of_y):
    loss_array = calculate_error(hypothesis, test_x_matrix, test_set_of_y)
    return loss_array[0]/len(test_set_of_y)#we only have 1 hypothesis at this point

#this function calculates errors
#hypothesis is 2-dim array
def calculate_error(hypothesis, x_matrix, set_of_y):
    loss_array = []
    for w in hypothesis:
        current = (np.dot(np.transpose(x_matrix[0:len(w)]),w))
        subtracted = np.subtract(current, set_of_y)
        loss_array.append(math.pow((np.linalg.norm(subtracted)),2))
    return loss_array


#k-fold algorithm, k=5:
def five_fold(training_set_x, training_set_y):
    #preparing X and splitting
    x_to_test = pol_reduction(training_set_x, 15)
    groups_of_x = np.split(x_to_test,5,1)
    errors = []
    for i in range(5):
        #dividing to validation set and training set
        validation_x = groups_of_x.pop(i)
        validation_y = training_set_y[40*i:40*(i+1)]
        train_x = np.column_stack(groups_of_x)
        train_y = training_set_y[0:40*i] + training_set_y[40*(i+1):]
        hypothesis_array = []
        #training all but i
        for j in range (1,16):
            hypothesis_array.append(least_square(train_x ,train_y,j))
        #validating on i
        chosen = test(hypothesis_array,validation_x, validation_y)
        errors.append(chosen)
        #returning i for next loop
        groups_of_x.insert(i,validation_x)
    return np.argmin(errors)





if __name__ == "__main__":
    #producing 3 sets
    X_training = cut_set(X,0,100)
    Y_training = cut_set(Y,0,100)
    X_validation = cut_set(X,100,200)
    Y_validation = cut_set(Y,100,200)
    X_test  = cut_set(X,200,300)
    Y_test  = cut_set(Y,200,300)
    #matrix for each set
    training_set_of_x = pol_reduction(X_training,15)
    validation_set_of_x = pol_reduction(X_validation,15)
    test_set_of_x = pol_reduction(X_test,15)
    #creating array of hypothesis by degree
    hypothesis_array = []
    for i in range (1,16):
        hypothesis_array.append(least_square(training_set_of_x ,Y_training,i))
    #choosing a hypothesis which minimizes least square:
    chosen= validate(hypothesis_array,validation_set_of_x, Y_validation)[0]
    #testing chosen:
    test_error = test([hypothesis_array[chosen]], test_set_of_x, Y_test)
    #running k-fold algorithm with k=5:
    X_k_fold = cut_set(X,0,200)
    Y_k_fold = cut_set(Y,0,200)
    print(five_fold(X_k_fold,Y_k_fold))


    pyplot.plot([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],validate(hypothesis_array,validation_set_of_x, Y_validation)[1],'kv')
    pyplot.show()