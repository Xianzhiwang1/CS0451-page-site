import typing
from typing import TypeVar
import random
import numpy as np

# def char_func(self,z):
#     return 1.0 if (z<0) else 0.0

class Perceptron:
    def __init__(self, w, history,maxiter):
        self.w = w
        self.history = history
        self.maxiter = maxiter

    def accuracy(X: np.array, y: np.array, w: np.array, b: np.array) -> int:
        return ( (X@w-b) * y >0).mean()

    
    def fit(self, X: np.array, y: np.array, b: int, maxiter: int) -> None:
        m,p = X.shape
        index = 0
        X_ = np.append(X, np.ones((X.shape[0],1)),1)
        w_ = np.append(self.w, -b) 
        while index <= maxiter and accuracy(X,y,w_,b) :
            a = random.randint(np.ndim(w))
            dotproduct = self.y_*np.dot(w_,X_)
            w_next = w_ + (dotproduct<0)*self.y*self.X
            index += 1 



        # fitting and training
        m,n = self.X.shape
        
    
    
    
   


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
















