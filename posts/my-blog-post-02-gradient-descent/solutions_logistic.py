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




    def fit(self, X: np.array, y: np.array, alpha: float, max_epochs: float) -> None:
        mu, nu = X.shape
        X_ = self.pad(X)
        W = np.random.randn(mu,nu)
        my_w = W[1,:]
        b = 1
        self.w_ = np.append(my_w, -b)
        history = []
        i = 0


        done = False
        prev_loss = np.inf


        while (not done) and (i <= max_epochs): 
            self.w_ -= alpha * self.gradient(self.w_, X_, y)                      # gradient step
            new_loss = empirical_risk(X_, y, logistic_loss, self.w_) # compute loss
            
            history.append(new_loss)
            # check if loss hasn't changed and terminate if so
            if np.isclose(new_loss, prev_loss):          
                done = True
            else:
                prev_loss = new_loss
            i += 1

            # update score
            self.score_history.append(self.score(X_, y))

        # self.w_ = w
        self.loss_history = history
        






    def predict(self, X):
        return (X@self.w_ > 0)*1

    def score(self, X, y) -> float:
        return 1 - self.loss(X,y) 


    # def loss(self, X,y):
    #     return 1-(predict(X, self.w_) == y).mean()
    def loss(self, X, y) -> float:
        return empirical_risk(X, y, logistic_loss, self.w_)



    def fit_stochastic(self, X: np.array, y: np.array, alpha: float, max_epochs: float, batch_size: int) -> None:
        n = X.shape[0]
        for j in np.arange(max_epochs):
                    
            order = np.arange(n)
            np.random.shuffle(order)

            for batch in np.array_split(order, n // batch_size + 1):
                x_batch = X[batch,:]
                y_batch = y[batch]
                grad = self.gradient(self.w_, x_batch, y_batch) 
                # perform the gradient step
                # ...

            mu, nu = X.shape
            X_ = self.pad(X)
            W = np.random.randn(mu,nu)
            my_w = W[1,:]
            b = 1
            self.w_ = np.append(my_w, -b)
            history = []
            i = 0


        done = False
        prev_loss = np.inf


        while (not done) and (i <= max_epochs): 
            self.w_ -= alpha * self.gradient(self.w_, X_, y)                      # gradient step
            new_loss = empirical_risk(X_, y, logistic_loss, self.w_) # compute loss
            
            history.append(new_loss)
            # check if loss hasn't changed and terminate if so
            if np.isclose(new_loss, prev_loss):          
                done = True
            else:
                prev_loss = new_loss
            i += 1

            # update score
            self.score_history.append(self.score(X_, y))

        # self.w_ = w
        self.loss_history = history



    def pad(self, X):
        return np.append(X, np.ones((X.shape[0], 1)), 1)




































