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



    def fit(self, X,y) -> None:
        # use gradient descent
        pass

    def loss(self, X,y):
        pass


    def fit_stochastic():
        pass

    def pad(X):
        return np.append(X, np.ones((X.shape[0], 1)), 1)




































