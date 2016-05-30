"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

AdaBoost classifier.

Author: Daniel Greenfeld & Inbal Lidan

"""
import numpy as np

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
        #initialize D as a vector of 1/m
        D = np.ones(len(y)) / len(y)
        for t in range (self.T):
             self.h[t] = self.WL(D,X,y)
             e_t = np.sum([D[k] for k in range(len(y)) if self.h[t].predict(X)[k]!=y[k]])
             self.w[t] = 0.5 * np.log(-1+1/e_t)
             D = [D[i]*np.exp(-1*self.w[t]*y[i]*self.h[t].predict(X)[i]) for i in range(len(y))]
             D /= np.sum(D)
             
    def predict(self, X):
        """
        Returns
        -------
        y_hat : a prediction vector for X
        """
        #calling the predict function for the decision stump object:
        h_x = [self.h[t].predict(X) for t in range(self.T)]
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