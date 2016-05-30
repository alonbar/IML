"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Skeleton for the decision tree classifier with real-values features.
Training algorithm: ID3

Author: Noga Zaslavsky
Date: April, 2016

"""
import numpy as np

def entropy(p):
    if p == 0 or p ==1:
        return 0
    else:
        return -p*np.log2(p)-(1-p)*np.log2(1-p)


class Node(object):
    """ A node in a real-valued decision tree.
        Set all the attributes properly for viewing the tree after training.
    """
    def __init__(self,leaf = True,left = None,right = None,samples = 0,feature = None,theta = 0.5,gain = 0,label = None):
        """
        Parameters
        ----------
        leaf : True if the node is a leaf, False otherwise
        left : left child
        right : right child
        samples : number of training samples that got to this node
        feature : a coordinate j in [d], where d is the dimension of x (only for internal nodes)
        theta : threshold over self.feature (only for internal nodes)
        gain : the gain of splitting the data according to 'x[self.feature] < self.theta ?'
        label : the label of the node, if it is a leaf
        """
        self.leaf = leaf
        self.left = left
        self.right = right
        self.samples = samples
        self.feature = feature
        self.theta = theta
        self.gain = gain
        self.label = label


class DecisionTree(object):
    """ A decision tree for bianry classification.
        Training method: ID3
    """

    def __init__(self,max_depth):
        self.root = None
        self.max_depth = max_depth

    def train(self, X, y):
        """
        Train this classifier over the sample (X,y)
        """
        # TODO - implement this method


    def ID3(self,X, y, A, depth):
        """
        Gorw a decision tree with the ID3 recursive method

        Parameters
        ----------
        X, y : sample
        A : array of d*m real features, A[j,:] row corresponds to thresholds over x_j
        depth : current depth of the tree

        Returns
        -------
        node : an instance of the class Node (can be either a root of a subtree or a leaf)
        """
        # TODO - implement this method


    @staticmethod
    def info_gain(X, y, A):
        """
        Parameters
        ----------
        X, y : sample
        A : array of m*d real features, A[:,j] corresponds to thresholds over x_j

        Returns
        -------
        gain : m*d array containing the gain for each feature
        """
        # TODO - implement this method


    def predict(self, X):
        """
        Returns
        -------
        y_hat : a prediction vector for X
        """
        # TODO - implement this method


    def error(self, X, y):
        """
        Returns
        -------
        the error of this classifier over the sample (X,y)
        """
        # TODO - implement this method
