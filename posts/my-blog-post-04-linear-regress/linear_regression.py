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
        # self.w_analytic = None # w_analytic
        self.new_loss = None
        self.loss_history = [] 
        self.score_history = []




    def y_hat(self, X, w):
        return X@w

    def loss_ell(self, y_hat, y): 
        return (y_hat - y)**2

    def empirical_risk(self, X, y, loss, w):
        y_hat = self.y_hat(X, w)
        return loss(y_hat, y).mean()

    def Big_L(self, X_, y) -> float:
        return np.linalg.norm(self.y_hat(X_, self.w)-y, ord = 2)

    def gradient(self, P, q):
        return  P@self.w - q 
    
    def fit_gradient(self, X_: np.array,  y: np.array, alpha = 0.00001, max_epochs = 1e4) -> None:
        # self.w = np.array([[0.99],
        #                   [0.99]])
        features = X_.shape[1]
        self.w = np.random.rand(features)
        self.score_history.append(self.score(X_,y))
        # initialization
        prev_loss = self.Big_L(X_ = X_, y = y) + 0.99 
        done = False
        i = 0
        P = X_.T@X_
        q = X_.T@y

        # while loop iteration
        while (not done) and (i <= max_epochs): 
            self.w = self.w - 2 * alpha * self.gradient(P, q) # gradient step
            self.new_loss = self.Big_L(X_ = X_, y = y) # compute loss
            # check if loss hasn't changed and terminate if so
            if np.isclose(self.new_loss, prev_loss, rtol=1e-05, atol=1e-08):          
                done = True
            
            prev_loss = self.new_loss

            # update before next loop
            self.loss_history.append(self.new_loss)
            self.score_history.append(self.score(X_,y))
            i += 1


    def fit_analytic(self, X_: np.array, y: np.array) -> None:
        w_hat = np.linalg.inv(X_.T@X_)@X_.T@y
        # self.w_analytic = w_hat
        self.w = w_hat



    def predict(self, X_) -> np.array:
        return X_ @ self.w




    
    def accuracy(self, X: np.array, y: np.array, w: np.array, ) -> float:
        pass



    def score(self, X_, y) -> float:
        y_hat = self.y_hat(X_, self.w)
        y_bar = y.mean()
        quotient = ((y_hat - y)**2).sum() / ((y_bar - y)**2).sum()
        return 1 - quotient 


    def pad(self, X):    
        return np.append(X, np.ones((X.shape[0],1)), 1)

    def myprint(self):
        print("it's working!")

        
