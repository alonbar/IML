"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Skeleton for the AdaBoost classifier.

Author: Noga Zaslavsky
Date: April, 2016

"""
import numpy as np
import sys
import ex6.ex6_tools
import math

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


if __name__ == "__main__":

    print ("main")
    a = AdaBoost(None, None)
    a.train()