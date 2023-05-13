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
        y_hat = X@w
        sum = 0
        n = X.shape[0]
        for i in range(n):
            summand = 1 * (y_hat[i] * y[i] > 0) 
            sum = sum + summand
        return sum/n
        # return ( (X @ w) * y >0).mean()

    
    def fit(self, X: np.array,  y: np.array, maxiter: int) -> None:
        # initialize a random initial weight vector 
        w = np.random.rand(X.shape[1], 1)
        b = 100
        index = 0
        accuracy = 0
        self.history.append(accuracy)

        # create X tilde, w tilde, y tilde
        X_ = np.append(X, np.ones((X.shape[0],1)),1)
        w_ = np.append(w, -b) 
        y_ = 2 * y-1

        # while loop
        while (index <= maxiter): # and (accuracy < 1):
            # pick a random index i \in [n]
            ndim = np.shape(X_)[0]
            # i = np.random.randint(w_shape+1)
            for i in range(ndim):
                x_i = X_[i,:]
                y_i = y_[i]
                # make sure it has the correct shape 
                w_ = w_.reshape(-1)
                # compute 
                dotprod = y_i * np.dot(w_, x_i) 
                w_next = w_ + (dotprod<0) * y_i * x_i
                # update before next loop
                w_ = w_next

            self.w_ = w_
            # append accuracy score to history
            accuracy = self.accuracy(X=X_, y=y_, w=w_)
            self.history.append(accuracy)
            index += 1 

        # # append accuracy score one last time
        # accuracy = self.accuracy(X=X_, y=y_, w=w_)
        # self.history.append(accuracy)

    


    
   


    def predict(self, X) -> np.array:
        X_ = np.append(X, np.ones((X.shape[0],1)),1)
        prod = X_ @ self.w_
        yhat = (prod > 0) * 1 
        return yhat 
        

    def score(self, X, y) -> float:
        return (self.predict(X) == y).mean()
        
        # X_ = np.append(X, np.ones((X.shape[0],1)),1)
        # return 2*( (X_ @ self.w_) * y >0).mean()

        
    # 1*(x>0)
    # 1*True = 1
    # 1*False = 0
    # if some_bool:
    #     w += update
    #     w += some_bool*update
    # done = (steps==maxiter) or (accuracy = 1)

    # # compute 
    # dotproduct = y_i *np.dot(w_,x_i)
    # w_next = w_ + (dotproduct<0) * y_i * x_i

    # # append accuracy score to history
    # accuracy = self.accuracy(X=X_, y=y_, w=w_)
    # self.history.append(accuracy)
    
    # # update before next loop
    # w_ = w_next
    # index += 1 













