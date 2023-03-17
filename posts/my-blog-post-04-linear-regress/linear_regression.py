import typing
from typing import TypeVar
from sklearn.linear_model import LogisticRegression
from mlxtend.plotting import plot_decision_regions
from scipy.optimize import minimize

import random
import numpy as np




# functions from https://middlebury-csci-0451.github.io/CSCI-0451/lecture-notes/gradient-descent.html


class LinearRegression:
    def __init__(self):
        self.y = None
        self.w = None # w_gradient
        self.w_analytic = None
        self.v = None
        self.new_loss = None
        self.history = [] 
        self.score_history = []



    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def y_hat(self, X, w):
        return X@w

    def loss_ell(self, y_hat, y): 
        return (y_hat - y)**2

    def empirical_risk(self, X, y, loss, w):
        y_hat = self.y_hat(X, w)
        return loss(y_hat, y).mean()

    def Big_L(self, X, w, y) -> float:
        return np.linalg.norm(self.y_hat(X,w)-y, ord = 2)

    def gradient(self, X, w, y):
        P = X.T@X
        q = X.T@y
        return  P@w - q 
    
    def fit_gradient(self, X: np.array,  y: np.array) -> None:
        # mu, nu = X.shape
        # print(mu,nu)
        # V = np.ones((mu, nu))
        # self.w = V[0,:]
        # self.v = np.array([1.0, 1.0])
        self.w = np.array([[1.0],
                          [1.0]])
        self.score_history = [ self.score(X,y) ]

        history = []
        max_epochs = 500 
        alpha = 0.0001

        prev_loss = self.Big_L(X = X, w = self.w, y = y) + 1

        done = False
        i = 0
        while (not done) and (i <= max_epochs): 
            self.w = self.w - 2 * alpha * self.gradient(X = X, w = self.w, y = y)                      # gradient step

            self.new_loss = self.Big_L(X = X, w = self.w, y = y) # compute loss
            # print(self.new_loss)
            history.append(self.new_loss)
            # check if loss hasn't changed and terminate if so
            if np.isclose(self.new_loss, prev_loss, rtol=1e-05, atol=1e-08):          
                done = True
            
            prev_loss = self.new_loss

            # update before next loop
            history.append(self.new_loss)
            self.score_history.append(self.score(X,y))
            i += 1
        print(self.w)
        self.history = history

    def fit_analytic(self, X: np.array, y: np.array) -> None:
        X_ = self.pad(X)
        # max_epochs = 100
        # alpha = 0.1
        # history = []
        w_hat = np.linalg.inv(X_.T@X_)@X_.T@y
        self.w_analytic = w_hat
        # while (not done) and (i <= max_epochs): 
        #     self.w -= 2 * alpha * self.gradient(self.w, X_, y)                      # gradient step
        #     new_loss = self.empirical_risk(X_, y, self.logistic_loss, self.w_) # compute loss
            
        #     history.append(new_loss)
        #     # check if loss hasn't changed and terminate if so
        #     if np.isclose(new_loss, prev_loss):          
        #         done = True
        #     else:
        #         prev_loss = new_loss
        #     i += 1



    def predict(self, X) -> np.array:
        pass




    
    def accuracy(self, X: np.array, y: np.array, w: np.array, ) -> float:
        pass



    def score(self, X, y) -> float:
        y_hat = self.y_hat(X, self.w)
        y_bar = y.mean()
        return 1 - (((y_hat - y)**2).sum() / ((y_bar - y)**2).sum())







    def pad(self, X):    
        return np.append(X, np.ones((X.shape[0],1)), 1)

    def myprint(self):
        print("it's working!")

        

        # mu, nu = X.shape
        # V = np.ones((mu,nu))
        # my_v = V[0,:]
        # mu = X.shape[0]
        # my_v = np.ones(mu)
        # b = 1
        # v = np.append(my_v, -b)
        # self.v = v
        # self.y = y 
        # print(self.v)

        # def matrmultiply(self, i: int, X_: np.array, v: np.array) -> np.array:
        #     # x_i = X[i,:]
        #     # x_col_j = X[:,j]
        #     # gram_matr = rbf_kernel(X_)
        #     km = self.kernel(X_, X_, **self.kernel_kwargs)
        #     row_i = km[i,:]
        #     return row_i@v
        
        #     myloss = 0
        #     ndim = np.shape(X)[0]
        #     km = self.kernel(X, X, **self.kernel_kwargs)

        #     for i in range(ndim):
        #         yi = y[i]
        #         row_i = km[i,:]
        #         y_hat_i = row_i@v 
        #         myloss += loss(y_hat_i, yi)
        #     return myloss /ndim

    # def empirical_risk_with_kernel(w: np.array, X_: np.array, y: np.array) -> float:
    #     ndim = np.shape(X_)[0]
    #     mysum = 0
    #     # i = np.random.randint(w_shape+1)
    #     for i in range(ndim):
    #         x_i = X_[i,:]
    #         yi = y[i]
    #         y_hat_i = np.dot(w, x_i)
    #         mysum += logistic_loss_derivative(y_hat_i, yi) * x_i

    #     return mysum / ndim 













