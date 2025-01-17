import typing
from typing import TypeVar
from sklearn.linear_model import LogisticRegression
from mlxtend.plotting import plot_decision_regions
from scipy.optimize import minimize
from matplotlib import pyplot as plt
plt.rcParams["figure.figsize"] = (8,4)
import numpy as np




# functions from https://middlebury-csci-0451.github.io/CSCI-0451/lecture-notes/gradient-descent.html


class KLR:
    def __init__(self, kernel, **kernel_kwargs):
        self.kernel = kernel
        self.kernel_kwargs = kernel_kwargs
        self.X_train = None
        self.y = None
        self.v = None


    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))


    def logistic_loss(self, y_hat, y): 
        return - y*np.log(self.sigmoid(y_hat)) - (1-y)*np.log(1-self.sigmoid(y_hat))


    # def matrmultiply(self, i: int, X_: np.array, v: np.array) -> np.array:
    #     # x_i = X[i,:]
    #     # x_col_j = X[:,j]
    #     # gram_matr = rbf_kernel(X_)
    #     km = self.kernel(X_, X_, **self.kernel_kwargs)
    #     row_i = km[i,:]

    #     return row_i@v


    # def empirical_risk_with_kernel(self, X: np.array, y: np.array, loss, v):
    #     myloss = 0
    #     ndim = np.shape(X)[0]
    #     km = self.kernel(X, X, **self.kernel_kwargs)


    #     for i in range(ndim):
    #         yi = y[i]
    #         row_i = km[i,:]
    #         y_hat_i = row_i@v 
    #         myloss += loss(y_hat_i, yi)
            
    #     return myloss /ndim

    def empirical_risk_with_kernel(self, X: np.array, y: np.array, loss, v):
        # myloss = 0
        # ndim = np.shape(X)[0]
        km = self.kernel(X, X, **self.kernel_kwargs)
        y_hat = km@v
            
        return loss(y_hat, y).mean() 


    def logistic_loss_derivative(self, y_hat, y) -> float:
        return self.sigmoid(y_hat) - y

    # def empirical_risk_with_kernel(w: np.array, X_: np.array, y: np.array) -> float:
    #     ndim = np.shape(X_)[0]
    #     mysum = 0
    #     # i = np.random.randint(w_shape+1)
    #     for i in range(ndim):
    #         x_i = X_[i,:]
    #         yi = y[i]
    #         y_hat_i = np.dot(w, x_i)
    #         mysum += logistic_loss_derivative(y_hat_i, yi) * x_i

    #     return mysum / ndim 
    def find_pars(self, X, y):
        
        n = X.shape[0]
        w0 = np.random.rand(n) # random initial guess
        
        # perform the minimization
        result = minimize(lambda v: self.empirical_risk_with_kernel(X = X, y = y, loss = self.logistic_loss,  v = v), 
                        x0 = w0) 
        
        # return the parameters
        return result.x

    
    def fit(self, X: np.array,  y: np.array) -> None:
        X_ = self.pad(X)
        # mu, nu = X.shape
        # V = np.ones((mu,nu))
        # my_v = V[0,:]
        mu = X.shape[0]
        v = np.random.rand(mu) 
        # my_v = np.ones(mu)
        # b = 1
        # v = np.append(my_v, -b)
        self.v = v
        self.y = y 
        # print(self.v)

        self.X_train = X_

        my_parameters = self.find_pars(self.X_train, self.y)

        self.v = my_parameters 
        # print(self.v)


    # def predict(self, X) -> np.array:
    #     X_ = self.pad(X) 
    #     km = self.kernel(X_, self.X_train, **self.kernel_kwargs) # km stands for kernel matrix
    #     innerProd = km@self.v
    #     y_hat = 1 * (innerProd > 0)

    #     return y_hat 

    def predict(self, X) -> np.array:
        X_ = self.pad(X)
        km = self.kernel(X_, self.X_train, **self.kernel_kwargs) # km stands for kernel matrix
        innerProd = km@self.v
        y_hat = 1 * (innerProd > 0)

        return y_hat 

    def accuracy(self, X: np.array, y: np.array, w: np.array, ) -> float:
        pass

    def score(self, X, y) -> float:
        pass

    def pad(self, X):    
        return np.append(X, np.ones((X.shape[0],1)), 1)

    def myprint(self, X, y, plot_function, value, width) -> None:
        fig, axarr = plt.subplots(1,2)
        axarr[0].scatter(X[:,0], X[:,1], c = y)
        axarr[0].set(xlabel = "Feature 0", ylabel = "Feature 1", title = f"artificial data created by {plot_function}")

        axarr[1].plot()
        plot_decision_regions(X, y, clf = self, 
                              feature_index=[0,1],
                              filler_feature_values={2: value, 3: value},
                              filler_feature_ranges={2: width, 3: width})
        yourpredict = self.predict(X)
        axarr[1].set(title = f"Accuracy = {(yourpredict == y).mean()}",
                            xlabel = "Feature 0", 
                            ylabel = "Feature 1")

        plt.tight_layout()

    def my_plot(self, X_cir, y_cir, plot_function) -> None:
        fig, axarr = plt.subplots(1,2)
        axarr[0].scatter(X_cir[:,0], X_cir[:,1], c = y_cir)
        axarr[0].set(xlabel = "Feature 1", ylabel = "Feature 2", title = f"artificial data created by {plot_function}")

        axarr[1].plot()
        plot_decision_regions(X_cir, y_cir, clf = self)
        yourpredict = self.predict(X_cir)
        axarr[1].set(title = f"Accuracy = {(yourpredict == y_cir).mean()}",
                            xlabel = "Feature 1", 
                            ylabel = "Feature 2")
        plt.tight_layout()


        
















