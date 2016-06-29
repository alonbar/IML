"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Running script for Ex6.

Author: Alon Bar
Date: April, 2016

"""
import ex6_tools
import matplotlib.pyplot as plot
import numpy as np
import math
import decision_tree


val_X = np.loadtxt('X_val.txt')
val_Y = np.loadtxt('y_val.txt')
X_train = np.loadtxt('X_train.txt')
y_train = np.loadtxt('y_train.txt')
X_test = np.loadtxt('X_test.txt')
y_test = np.loadtxt('y_test.txt')

class AdaBoost(object):

    def __init__(self, WL, T):
        """
        Parameters
        ----------
        WL : the class of the base weak learner
        T : the number of base learners to learn
        """
        self.WL = WL
        self.T = T
        self.h = [None]*T     # list of base learners
        self.w = np.zeros(T)  # weights


    def train(self, X, y):
        """
        Train this classifier over the sample (X,y)
        """
        D = np.ones(len(X)) / len(X)
        for t in range(self.T):
            self.h[t] = self.WL(D,X,y)
            L_d = []
            for i in range(len(y)):
                if self.h[t].predict(X)[i] != y[i]:
                    L_d.append(D[i])
            epsilon_t = np.sum(L_d)

            self.w[t] = 0.5 * np.log((1/epsilon_t) - 1)
            D = [D[i]*np.exp(-1*self.w[t]*y[i]*self.h[t].predict(X)[i]) for i in range(len(y))]
            D /= np.sum(D)

    def predict(self, X):
        """
        Returns
        -------
        y_hat : a prediction vector for X
        """
        h_x = []
        for t in range(self.T):
            h_x += [self.h[t].predict(X)]
        h_x = np.column_stack(h_x)
        #calculate sign(<h_x,w>)
        y_hat = np.array(np.sign(np.dot(h_x,self.w)))
        y_hat[y_hat==0] = 1
        return y_hat


    def error(self, X, y):
        """
        Returns
        -------
        the error of this classifier over the sample (X,y)
        """
        y_hat = self.predict(X)
        return np.sum(1 for i in range(len(y)) if y[i]!=y_hat[i]) / len(y)



def Q3(): # AdaBoost
    T = []
    T.append(1)
    for i in range(1,41):
        T.append(i*5)
    H = []
    for t in T:
        H.append(AdaBoost(ex6_tools.DecisionStump,t))

    data = np.loadtxt('X_train.txt')
    tag = np.loadtxt('y_train.txt')

    for i in range(len(T)):
        H[i].train(data,tag)

    T_boundries = [1,5,10,50,100,200]
    for i in range(len(T_boundries)):
        ex6_tools.decision_boundaries(H[i],data,tag,"data")

    plot.show()

    training_error = []
    for h in H:
        training_error += [h.error(data,tag)]

    val_tag = np.loadtxt('y_val.txt')
    val_data = np.loadtxt('X_val.txt')
    val_data = np.row_stack(val_data)

    val_error = []
    for h in H:
        val_error += [h.error(val_data,val_tag)]

    plot.figure(2)

    plot.scatter(T, training_error)
    plot.scatter(T, val_error, c='r')
    plot.title('blue is Traning error , red is Validation error')
    plot.show()

    print ("T that minimizes the validation error is: ",np.argmin(val_error))
    plot.figure(3)
    ex6_tools.decision_boundaries(H[np.argmin(val_error)],data,tag,"data")
    plot.title('min validation error training boundaries')



    return

def Q4(): # decision trees

    max_depths = range(1, 13)
    decision_trees = [None]*len(max_depths)
    valid_error = np.zeros(len(max_depths))
    train_error = np.zeros(len(max_depths))
    plot.figure()
    plot.title('boundries')
    # Train decision tree classiers over the training data, with max_depth = 1,...,12.
    # Plot the decisions of each learned tree
    for depth in max_depths:
        tree = decision_tree
        decision_trees[depth-1] = tree.DecisionTree(depth)
        decision_trees[depth-1].train(X_train, y_train)
        train_error[depth-1] = decision_trees[depth-1].error(X_train, y_train)
        valid_error[depth-1] = decision_trees[depth-1].error(val_X, val_Y)
        plot.subplot(3, len(max_depths) / 3, depth)
        ex6_tools.decision_boundaries(decision_trees[depth-1], X_train, y_train, 'max_depth=' + str(depth))
    # Plot the training error and validation error as a function of max_depth.
    plot.figure()
    plot.plot(max_depths, valid_error, color='red', label='Validation Error')
    plot.plot(max_depths, train_error, color='blue', label='Train Error')
    plot.title('Selection of trees')
    plot.legend()
    plot.xlabel('depth'), plot.ylabel('error')

    # Show results of the optimal(w.r.t validation data) T
    optimal_depth = np.argmin(valid_error) + 1
    optimal_tree = decision_trees[optimal_depth-1]
    testing_error = optimal_tree.error(X_test, y_test)
    print ("opt_max_depth: ", optimal_depth)
    print ("training   error: ", train_error[optimal_depth-1])
    print ("validation error: ", valid_error[optimal_depth-1])
    print ("test       error: ", testing_error)

    # Visualize the best tree.
    ex6_tools.view_dtree(optimal_tree, filename='dtree_best')
    # Show figures
    plot.show(block=True)
    return

def Q5(): # kNN
    # TODO - implement this function
    return

def Q6(): # Republican or Democrat?
    # TODO - implement this function
    return

if __name__ == '__main__':
    # TODO - run your code for questions 3-6
    # Q3()
    Q4()
    pass