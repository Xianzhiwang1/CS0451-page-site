import typing
from typing import TypeVar
import random
import numpy as np

# def char_func(self,z):
#     return 1.0 if (z<0) else 0.0

class AuditBias:
    def __init__(self):
        self.history = [] 
        self.w_ = None    

    def accuracy(self, X: np.array, y: np.array, w: np.array, ) -> float:
        pass




