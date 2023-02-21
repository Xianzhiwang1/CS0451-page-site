import typing
from typing import TypeVar
import random
import numpy as np

# def char_func(self,z):
#     return 1.0 if (z<0) else 0.0

class Perceptron:
    def __init__(self, history, maxiter):
        self.history = history
        self.maxiter = maxiter

    def accuracy(self, X: np.array, y: np.array, w: np.array, ) -> float:
        return ( (X @ w) * y >0).mean()

    
    def fit(self, X: np.array, w: np.array, y: np.array, b: int, maxiter: int) -> None:
        m,p = X.shape
        index = 0
        X_ = np.append(X, np.ones((X.shape[0],1)),1)
        w_ = np.append(w, -b) 
        y_ = 2 * y-1
        while (index <= maxiter) and (self.accuracy(X = X_ ,y = y_ ,w = w_ ) < 1):
            # initialize a random initial weight vector 

            # pick a random index i \in [n]
            i = random.randint(np.ndim(x_i))
            x_i = X_[i,:]
            y_i = y_[i]

            
            # compute 
            dotproduct = y_i *np.dot(w_,x_i)
            w_next = w_ + (dotproduct<0) * y_i * x_i
            
            # update before next loop
            w_ = w_next
            index += 1 


    
   


    def predict(self, X):
        pass
        
    def score(X,y):
        pass

    def myprint(self):
        print("it's running around!")

        
    # 1*(x>0)
    # 1*True = 1
    # 1*False = 0
    # if some_bool:
    #     w += update
    #     w += some_bool*update
    # done = (steps==maxiter) or (accuracy = 1)
















