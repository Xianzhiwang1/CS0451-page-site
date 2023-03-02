import typing
from typing import TypeVar
import random
import numpy as np


class KLR:
    def __init__(self, kernel, gamma):
        self.kernel = kernel
        self.gamma = gamma

    def accuracy(self, X: np.array, y: np.array, w: np.array, ) -> float:
        pass

    
    def fit(self, X: np.array,  y: np.array, maxiter: int) -> None:
        pass

    

    def predict(self, X) -> np.array:
        pass


    def score(self, X, y) -> float:
        pass
        
        # X_ = np.append(X, np.ones((X.shape[0],1)),1)
        # return 2*( (X_ @ self.w_) * y >0).mean()

    def myprint(self):
        print("it's working!")

        
















