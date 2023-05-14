import numpy as np
from sklearn.datasets import make_blobs
from matplotlib.patches import Patch
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.linear_model import LogisticRegression
le = LabelEncoder()


# for plotting decision region
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from mlxtend.plotting import plot_decision_regions

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score


# Logistic Regression and Newton Raphson 
class Newton_Raphson():
    def __init__(self, *kernel, **kernel_kwargs):
        self.kernel = kernel
        self.kernel_kwargs = kernel_kwargs
        self.beta_old = np.random.rand(3,1)
        self.beta = np.random.rand(3,1)
        self.alpha = 0.5   
        self.reg = True

    def sigmoid(self, x: np.array):
        return 1/(1+np.exp(-x))

    def prob(self, X: np.array, beta: np.array):
        '''
        prob stands for probability
        '''
        return np.array(self.sigmoid(X.dot(beta)))
        # return self.sigmoid(X.T @ beta)
        # return self.sigmoid(X @ beta)

    def Var(self, p: np.array):
        '''
        Var stands for Variance 
        '''
        on_diag = p*(1-p)
        d = on_diag.reshape(-1)
        return np.diag(d)

    def Hessian(self, X: np.array, beta: np.array):
        '''
        print(f"X.T.shape {X.T.shape}")
        print(f"X.shape {X.shape}")
        print(f"VAR.shape {V.shape}")
        print(XV.shape)
        '''
        V = self.Var(self.prob(X,beta))
        XV = X.T @ V
        # result = np.dot(XV, X)
        result = XV @ X

        return result

    def update(self, y, X, beta):
        '''
        print(X.T.shape)
        print(y.shape)
        print(self.prob(X,beta).shape)
        '''
        grad = X.T @ (y-self.prob(X,beta))
        # reguliazation
        if (self.reg):
            hessian = self.Hessian(X,beta) + np.eye(beta.shape[0])
            step = np.linalg.inv(hessian) @ grad
        # no reguliazation
        else:
            hessian = self.Hessian(X,beta)
            step = np.linalg.inv(hessian) @ grad
        # print(f"step: {step.shape}")
        step = self.alpha * step
        beta = beta + step 
        return beta

    def regress(self, y, X, max_iters = 1e1, tol=1e-8, converged=False):
        X = self.patek(X)
        self.beta_old = np.ones((X.shape[1],1))
        # self.beta = np.array([-5.4e-02,  6.8e-02, 2.3e-2]) 
        self.beta = np.random.rand(X.shape[1],1)
        self.beta = self.beta.reshape(-1,1)
        self.beta_old = self.beta_old.reshape(-1,1)
        # print(f"learning rate is: {self.alpha}")
        # print(f"Regularization is: {self.reg}")
        
        iter_count = 0 
        while not converged and (iter_count<max_iters):
            iter_count += 1
            # print(f"self.beta {self.beta.shape}")
            # print(f"self.beta_old {self.beta_old.shape}")

            self.beta = self.update(y, X, self.beta)
            if (iter_count % 10 == 0):
                print(f"number of iteration: {iter_count}")
                # print(f"beta: {self.beta}")

            if not np.any(np.abs(self.beta_old - self.beta)>tol):
                converged = True
                print(f"Converged with {iter_count} iterations")
                print(f"The beta we end up with is: {self.beta}")
            self.beta_old = self.beta


        # self.beta = self.beta.to_numpy()

    def predict(self, X):
        X = self.patek(X)
        innerProd = X @ self.beta
        y_hat = 1 * (innerProd > 0)
        return y_hat

    def score(self,X,y):
        y = y.reshape(-1)
        mypredict = self.predict(X)
        mypredict = mypredict.reshape(-1)
        myscore = 1 * (mypredict == y) 
        return myscore.mean()

    def simple_plot(self, model, X, y, F1,F2):
        y = y.reshape(-1)
        plot_decision_regions(X, y, clf=model)
        self.mypredict = model.predict(X)
        score = (self.mypredict==y).mean()
        title = plt.gca().set(title=f"Accuracy={round(score,3)} using {model}",
        # title = plt.gca().set(title=f"Accuracy using {model}",
                            xlabel=F1,
                            ylabel=F2)

    def bare_bone_plot(self, X, y, size_1, size_2):
        plt.rcParams["figure.figsize"] = (size_1,size_2)
        mypredict = self.predict(X)
        score = (mypredict == y).mean()
        print(f"the weight beta is: {self.beta}")
        a_0 = self.beta[0][0]
        a_1 = self.beta[1][0]
        a_2 = self.beta[2][0]
        fig = plt.scatter(X[:,0], X[:,1], c = y)
        xlab = plt.xlabel("Feature 1")
        ylab = plt.ylabel("Feature 2")
        f1 = np.linspace(3.5, 4.5, 501)
        p = plt.plot(f1,  -(a_2/a_1) - (a_0/a_1)*f1, color = "black")
        title = plt.gca().set_title(f"score is: {round(score,3)}")
    


    def helper_plot(self, X, y, subplot, label, F1,F2):
        '''
        Helper method for big_plot
        '''
        score = self.score(X,y) 
        a_0 = self.beta[0][0]
        a_1 = self.beta[1][0]
        a_2 = self.beta[2][0]
        fig = subplot.scatter(X[:,0], X[:,1], c = y)
        subplot.set(xlabel=F1, ylabel=F2, title=f"score: {round(score, 3)} using {label} data")
        # the line
        f1 = np.linspace(3.5, 14.5, 501)
        p = subplot.plot(f1,  -(a_2/a_1) - (a_0/a_1)*f1, color = "black")

    def big_plot(self, X_train, y_train, X_validate, y_validate, X_test, y_test, size_1, size_2, F1, F2):
        plt.rcParams["figure.figsize"] = (size_1,size_2)
        fig, axarr = plt.subplots(1,3)
        self.helper_plot(X_train, y_train, axarr[0], "training", F1, F2)
        self.helper_plot(X_validate, y_validate, axarr[1], "validation", F1, F2)
        self.helper_plot(X_test, y_test, axarr[2], "testing", F1, F2)
        plt.tight_layout()



    def patek(self, X: np.array) -> np.array:
        '''
        Certified pre-owned bust-down patek function for padding
        '''
        return np.append(X, np.ones((X.shape[0],1)), 1)







