import typing
from typing import TypeVar
import random
import numpy as np

# def char_func(self,z):
#     return 1.0 if (z<0) else 0.0

class Perceptron:
    def __init__(self):
        self.history = [] 
        self.w_ = None    

    def accuracy(self, X: np.array, y: np.array, w: np.array, ) -> float:
        return ( (X @ w) * y >0).mean()

    
    def fit(self, X: np.array,  y: np.array, maxiter: int) -> None:
        # initialize a random initial weight vector 
        mu, nu = X.shape
        W = np.random.randn(mu,nu)
        w = W[1,:]
        b = 10
        index = 0

        # create X tilde, w tilde, y tilde
        X_ = np.append(X, np.ones((X.shape[0],1)),1)
        w_ = np.append(w, -b) 
        y_ = 2 * y-1

        # while loop
        while (index <= maxiter) and (self.accuracy(X = X_ ,y = y_ ,w = w_ ) < 1):

            # pick a random index i \in [n]
            w_shape = np.shape(w)[0]
            i = np.random.randint(w_shape+1)
            x_i = X_[i,:]
            y_i = y_[i]

            
            # compute 
            dotproduct = y_i *np.dot(w_,x_i)
            w_next = w_ + (dotproduct<0) * y_i * x_i

            # append accuracy score to history
            accuracy = self.accuracy(X=X_, y=y_, w=w_)
            self.history.append(accuracy)
            
            
            # update before next loop
            w_ = w_next
            index += 1 
        self.w_ = w_
        # append accuracy score one last time
        accuracy = self.accuracy(X=X_, y=y_, w=w_)
        self.history.append(accuracy)

    


    
   


    def predict(self, X) -> np.array:
        X_ = np.append(X, np.ones((X.shape[0],1)),1)
        prod = X_ @ self.w_

        yhat = (prod >= 0) * 1 
        return yhat 
        

    def score(self, X, y) -> float:
        X_ = np.append(X, np.ones((X.shape[0],1)),1)
        return ( (X_ @ self.w_) * y >0).mean()

    def myprint(self):
        print("it's working!")

        
    # 1*(x>0)
    # 1*True = 1
    # 1*False = 0
    # if some_bool:
    #     w += update
    #     w += some_bool*update
    # done = (steps==maxiter) or (accuracy = 1)
















