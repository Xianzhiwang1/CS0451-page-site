import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.datasets import make_blobs

np.random.seed(12345)
n=100
p_features = 3

X, y = make_blobs(n_samples = 100, n_features=p_features-1, centers=[ (-1.7,-1.7), (1.7,1.7) ])

fig=plt.scatter(X[:,0], X[:,1], c=y)
xlab=plt.xlabel("feature 1")
ylab=plt.ylabel("feature 2")


from perceptron import Perceptron
w = 1
history = 2
b = 10
p = Perceptron(history = history, maxiter = 100)
p.myprint()
p.fit(X,y,b, maxiter=10)