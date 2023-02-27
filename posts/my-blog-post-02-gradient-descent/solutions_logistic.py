import typing
from typing import TypeVar
import random
import numpy as np
from scipy.optimize import minimize
np.seterr(all='ignore') 

# add a constant feature to the feature matrix
# X_ = np.append(X, np.ones((X.shape[0], 1)), 1)

# functions from https://middlebury-csci-0451.github.io/CSCI-0451/lecture-notes/gradient-descent.html
def predict(X, w):
    return X@w

def sigmoid(z):
    return 1 / (1 + np.exp(-z))




def logistic_loss(y_hat, y): 
    return -y*np.log(sigmoid(y_hat)) - (1-y)*np.log(1-sigmoid(y_hat))

def empirical_risk(X, y, loss, w):
    y_hat = predict(X, w)
    return loss(y_hat, y).mean()



class LogisticRegression():
    def __init__(self) -> None:
        self.w_ = None
        self.loss_history = []
        self.score_history = []

    def logistic_loss_derivative(self, y_hat, y) -> float:
        return sigmoid(y_hat) - y

    def gradient(self, w: np.array, X_: np.array, y: np.array) -> float:
        ndim = np.shape(X_)[0]
        mysum = 0
        # i = np.random.randint(w_shape+1)
        for i in range(ndim):
            x_i = X_[i,:]
            yi = y[i]
            y_hat_i = np.dot(w, x_i)
            mysum += self.logistic_loss_derivative(y_hat_i, yi) * x_i

        return mysum / ndim 




    def fit(self, X_: np.array, y: np.array, alpha: float, max_epochs: float) -> None:
        # use gradient descent
        mu, nu = X_.shape
        W = np.random.randn(mu,nu)
        w = W[1,:]


        done = False
        prev_loss = np.inf
        history = []


        while not done: 
            w -= alpha * self.gradient(w, X_, y)                      # gradient step
            new_loss = empirical_risk(X_, y, logistic_loss, w) # compute loss
            
            history.append(new_loss)
            # check if loss hasn't changed and terminate if so
            if np.isclose(new_loss, prev_loss):          
                done = True
            else:
                prev_loss = new_loss

    def predict(self, X):
        pass

    def score(self, X):
        pass




    def loss(self, X,y):
        pass


    def fit_stochastic(self):
        pass

    def pad(self, X):
        return np.append(X, np.ones((X.shape[0], 1)), 1)




































