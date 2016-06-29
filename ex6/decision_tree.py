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
        matrix = np.zeros((X.shape[0] - 1, X.shape[1]))
        sortedX = np.sort(X, axis=0)
        for i in range(sortedX.shape[0] - 1):
            matrix[i, :] = (sortedX[i, :] + sortedX[i + 1, :]) / 2.0
        self.root = self.ID3(X, y, matrix, 0)


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
        y_sum_1 = np.sum(y == 1)
        y_sum_minus_1 =np.sum(y == -1)
        # if all examples labeled -1
        if y_sum_minus_1 == len(y):
            return Node(leaf=True, samples=X.shape[0], label=-1)
        if y_sum_1 == len(y):
            return Node(leaf=True, samples=X.shape[0], label=1)
        # if A is empty or maxDepth was reached
        if (A.size == 0 or depth == self.max_depth):
            # return a leaf with majority label
            u, indices = np.unique(y, return_inverse=True)
            return Node(leaf=True, samples=X.shape[0], label=u[np.argmax(np.bincount(indices))])
        # calc gain and arg_max
        G = self.info_gain(X, y, A)
        index_max = np.unravel_index(np.argmax(G), G.shape)
        j = index_max[1]
        best_theta = A[index_max]
        best_gain = G[index_max]
        num_of_samples = X.shape[0]
        # update X and A
        right_samples = X[:, j] >= best_theta
        left_samples = X[:, j] < best_theta
        A[index_max] = float("-inf")
        # features = np.ones(X.shape[1], dtype=bool)
        # features[j] = False  # Delete j'th feature
        T1 = self.ID3(X[right_samples, :], y[right_samples], A, depth+1)
        T2 = self.ID3(X[left_samples, :], y[left_samples], A, depth+1)
        return Node(leaf=False, left=T2, right=T1, samples=num_of_samples, feature=j, theta=best_theta, gain=best_gain)


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
        gain = np.zeros(A.shape)
        # positive = (y == 1)
        sign_array = []
        for i, flag in enumerate(y):
            if y[i] == 1:
                sign_array.append(True)
            else:
                sign_array.append(False)
        p_y = np.sum(sign_array) / float(len(X))  # probability of y=1 over S
        before = entropy(p_y)
        # check if A is a vector or a matrix
        # colRange = 0
        # if A.ndim == 2:
        #     colRange = A.shape[1]
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                p_x_j = np.sum(X[:, j] < A[i, j]) / float(len(X))  # probability of x_j<theta_ij over S
                if p_x_j == 1 or p_x_j == 0:
                    # unreliable threshold
                    gain[i, j] = -1
                    continue
                p_y_and_x_j = np.sum(X[sign_array,j] < A[i, j]) / float(len(X))
                p_y_Given_x_j = p_y_and_x_j / p_x_j
                p_y_and_not_x_j = np.sum(X[sign_array, j] >= A[i, j]) / float(len(X))
                p_y_Given_not_x_j = p_y_and_not_x_j / (1-p_x_j)
                after = p_x_j*entropy(p_y_Given_x_j) + (1-p_x_j)*entropy(p_y_Given_not_x_j)
                gain[i, j] = before - after
        return gain

    def predict(self, X):
        """
        Returns
        -------
        y_hat : a prediction vector for X
        """
        y_hat = np.zeros(X.shape[0])
        for i in range(len(X)):
            curr = self.root
            while not curr.leaf:
                if X[i, curr.feature] < curr.theta:
                    curr = curr.left
                else:
                    curr = curr.right
            y_hat[i] = curr.label
        return y_hat


    def error(self, X, y):
        """
        Returns
        -------
        the error of this classifier over the sample (X,y)
        """
        # TODO - implement this method
        y_hat = self.predict(X)
        diff = np.multiply(y_hat, y)
        return (np.sum(diff == -1)) / float(len(X))














# import numpy as np
#
#
# def entropy(p):
#     if p == 0 or p == 1:
#         return 0
#     else:
#         if p > 1:
#             print (p)
#         return -p*np.log2(p)-(1-p)*np.log2(1-p)
#
#
# class Node(object):
#     """ A node in a real-valued decision tree.
#         Set all the attributes properly for viewing the tree after training.
#     """
#     def __init__(self, leaf=True, left=None, right=None, samples=0, feature=None, theta=0.5, gain=0, label=None):
#         """
#         Parameters
#         ----------
#         leaf : True if the node is a leaf, False otherwise
#         left : left child
#         right : right child
#         samples : number of training samples that got to this node
#         feature : a coordinate j in [d], where d is the dimension of x (only for internal nodes)
#         theta : threshold over self.feature (only for internal nodes)
#         gain : the gain of splitting the data according to 'x[self.feature] < self.theta ?'
#         label : the label of the node, if it is a leaf
#         """
#         self.leaf = leaf
#         self.left = left
#         self.right = right
#         self.samples = samples
#         self.feature = feature
#         self.theta = theta
#         self.gain = gain
#         self.label = label
#
#
# class DecisionTree(object):
#     """ A decision tree for binary classification.
#         Training method: ID3
#     """
#
#     def __init__(self, max_depth):
#         self.root = None
#         self.max_depth = max_depth
#
#     def train(self, X, y):
#         """
#         Train this classifier over the sample (X,y)
#         """
#         A = self.calcThresholds(X)
#         self.root = self.ID3(X, y, A, 0)  # Start with depth=0
#
#     def ID3(self, X, y, A, depth):
#         """
#         Grow a decision tree with the ID3 recursive method
#
#         Parameters
#         ----------
#         X, y : sample
#         A : array of d*m real features, A[j,:] row corresponds to thresholds over x_j
#         depth : current depth of the tree
#
#         Returns
#         -------
#         node : an instance of the class Node (can be either a root of a subtree or a leaf)
#         """
#
#         # if all examples labeled 1
#         if np.sum(y == 1) == len(y):
#             return Node(leaf=True, samples=X.shape[0], label=1)
#         # if all examples labeled -1
#         if np.sum(y == -1) == len(y):
#             return Node(leaf=True, samples=X.shape[0], label=-1)
#         # if A is empty or maxDepth was reached
#         if A.size == 0 or depth == self.max_depth:
#             # return a leaf with majority label
#             u, indices = np.unique(y, return_inverse=True)
#             return Node(leaf=True, samples=X.shape[0], label=u[np.argmax(np.bincount(indices))])
#         # calc gain and arg_max
#         G = self.info_gain(X, y, A)
#         max_ind = np.unravel_index(np.argmax(G), G.shape)
#         j = max_ind[1]
#         best_theta = A[max_ind]
#         best_gain = G[max_ind]
#         num_of_samples = X.shape[0]
#         # update X and A
#         right_samples = X[:, j] >= best_theta
#         left_samples = X[:, j] < best_theta
#         A[max_ind] = float("-inf")
#         # features = np.ones(X.shape[1], dtype=bool)
#         # features[j] = False  # Delete j'th feature
#         T1 = self.ID3(X[right_samples, :], y[right_samples], A, depth+1)
#         T2 = self.ID3(X[left_samples, :], y[left_samples], A, depth+1)
#         return Node(leaf=False, left=T2, right=T1, samples=num_of_samples, feature=j, theta=best_theta, gain=best_gain)
#
#     @staticmethod
#     def calcThresholds(X):
#         TH = np.zeros((X.shape[0] - 1, X.shape[1]))
#         sortedX = np.sort(X, axis=0)
#         for i in range(sortedX.shape[0] - 1):
#             TH[i, :] = (sortedX[i, :] + sortedX[i + 1, :]) / 2.0
#         return TH
#
#     @staticmethod
#     def info_gain(X, y, A):
#         """
#         Parameters
#         ----------
#         X, y : sample
#         A : array of m*d real features, A[:,j] corresponds to thresholds over x_j
#
#         Returns
#         -------
#         gain : m*d array containing the gain for each feature
#         """
#         gain = np.zeros(A.shape)
#         positive = (y == 1)
#         p_y = np.sum(positive) / float(len(X))  # probability of y=1 over S
#         before = entropy(p_y)
#         # check if A is a vector or a matrix
#         # colRange = 0
#         # if A.ndim == 2:
#         #     colRange = A.shape[1]
#         for i in range(A.shape[0]):
#             for j in range(A.shape[1]):
#                 p_x_j = np.sum(X[:, j] < A[i, j]) / float(len(X))  # probability of x_j<theta_ij over S
#                 if p_x_j == 1 or p_x_j == 0:
#                     # unreliable threshold
#                     gain[i, j] = -1
#                     continue
#                 p_y_and_x_j = np.sum(X[positive,j] < A[i, j]) / float(len(X))
#                 p_y_Given_x_j = p_y_and_x_j / p_x_j
#                 p_y_and_not_x_j = np.sum(X[positive, j] >= A[i, j]) / float(len(X))
#                 p_y_Given_not_x_j = p_y_and_not_x_j / (1-p_x_j)
#                 after = p_x_j*entropy(p_y_Given_x_j) + (1-p_x_j)*entropy(p_y_Given_not_x_j)
#                 gain[i, j] = before - after
#         return gain
#
#     def predict(self, X):
#         """
#         Returns
#         -------
#         y_hat : a prediction vector for X
#         """
#         y_hat = np.zeros(X.shape[0])
#         for i in range(len(X)):
#             curr = self.root
#             while not curr.leaf:
#                 if X[i, curr.feature] < curr.theta:
#                     curr = curr.left
#                 else:
#                     curr = curr.right
#             y_hat[i] = curr.label
#         return y_hat
#
#     def error(self, X, y):
#         """
#         Returns
#         -------
#         the error of this classifier over the sample (X,y)
#         """
#         y_hat = self.predict(X)
#         diff = np.multiply(y_hat, y)
#         return (np.sum(diff == -1)) / float(len(X))













