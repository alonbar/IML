"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Running script for Ex6.

Author: Inbal LIdan & Daniel Greenfeld
Date: April, 2016

"""
import inbal_adaboost
import ex6_tools

import numpy as np
import matplotlib.pyplot as plt


data = np.loadtxt('X_train.txt')
data = np.row_stack(data)
tag = np.loadtxt('y_train.txt').tolist()
val_data = np.loadtxt('X_val.txt')
val_tag = np.loadtxt('y_val.txt').tolist()
val_data = np.row_stack(val_data)
test_data = np.loadtxt('X_test.txt')
test_tag = np.loadtxt('y_test.txt').tolist()
test_data = np.row_stack(test_data)

def Q3(): # AdaBoost
    # TODO - implement this function
    T = [5 * i for i in range(1,41)]
    H = [inbal_adaboost.AdaBoost(ex6_tools.DecisionStump,t) for t in T]
    T1 = [1,5,10,50,100,200]
    for i in range(len(T)):
        H[i].train(data,tag)
    for i in range(len(T1)):
        ex6_tools.decision_boundaries(H[i],data,tag,"data")
        # decision_boundaries(classifier, X, y, title_str='', weights=None):
    plt.show()

    training_error = [h.error(data,tag) for h in H]
    val_error = [h.error(val_data,val_tag) for h in H]
    plt.figure(2)

    plt.scatter(T,training_error)
    plt.scatter(T,val_error,c='r')
    plt.title('Traning error in blue, Validation error in red')
    plt.show()

    T_min = np.argmin(val_error)
    print ("T that minimizes the validation error is: ",T[T_min])
    plt.figure(3)
    ex6_tools.decision_boundaries(H[T_min],data,tag,"data")
    plt.title('min validation error training boundaries')

def Q4(): # decision trees
    tree = decision_tree.DecisionTree(4)
    tree.train(data,tag)
    ex6_tools.view_dtree(tree)

def Q5(): # kNN

    K = [1,3,10,100,200,500]
    H = [nearest_neighbors.kNN(k) for k in K]
    for h in H:
        h.train(data,tag)

    for i in range(len(K)):
        ex6_tools.decision_boundaries(H[i],data,tag,2,3,i,i)
    plt.show()

    training_error = [h.error(data,tag) for h in H]
    val_error = [h.error(val_data,val_tag) for h in H]
    plt.figure(2)

    plt.scatter(T,training_error)
    plt.scatter(T,val_error,c='r')
    plt.title('Traning error in blue, Validation error in red')
    plt.show()

    T_min = np.argmin(val_error)
    print ("T that minimizes the validation error is: ",[T_min])
    plt.figure(3)
    ex6_tools.decision_boundaries(H[T_min],data,tag,1,1,0)
    plt.title('min validation error training boundaries')


def Q6(): # Republican or Democrat?
    # TODO - implement this function
    return

if __name__ == '__main__':
    # TODO - run your code for questions 3-6
    #print(data)
    # Q4()
    # Q5()
    Q3()
