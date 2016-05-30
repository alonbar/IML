"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Running script for Ex6.

Author: Alon Bar
Date: April, 2016

"""
import adaboost
import ex6_tools
import matplotlib.pyplot as plt
import numpy as np


def Q3(): # AdaBoost
    T = []
    T.append(1)
    for i in range(1,41):
        T.append(i*5)
    H = []
    for t in T:
        H.append(adaboost.AdaBoost(ex6_tools.DecisionStump,t))

    data = np.loadtxt('X_train.txt')
    tag = np.loadtxt('y_train.txt')

    for i in range(len(T)):
        H[i].train(data,tag)

    T_boundries = [1,5,10,50,100,200]
    for i in range(len(T_boundries)):
        ex6_tools.decision_boundaries(H[i],data,tag,"data")

    plt.show()

    training_error = []
    for h in H:
        training_error += [h.error(data,tag)]

    val_tag = np.loadtxt('y_val.txt')
    val_data = np.loadtxt('X_val.txt')
    val_data = np.row_stack(val_data)

    val_error = []
    for h in H:
        val_error += [h.error(val_data,val_tag)]

    plt.figure(2)

    plt.scatter(T,training_error)
    plt.scatter(T,val_error,c='r')
    plt.title('blue is Traning error , red is Validation error')
    plt.show()

    print ("T that minimizes the validation error is: ",np.argmin(val_error))
    plt.figure(3)
    ex6_tools.decision_boundaries(H[np.argmin(val_error)],data,tag,"data")
    plt.title('min validation error training boundaries')



    return

def Q4(): # decision trees
    # TODO - implement this function
    return

def Q5(): # kNN
    # TODO - implement this function
    return

def Q6(): # Republican or Democrat?
    # TODO - implement this function
    return

if __name__ == '__main__':
    # TODO - run your code for questions 3-6
    Q3()

    pass