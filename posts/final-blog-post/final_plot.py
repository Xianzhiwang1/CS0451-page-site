
import numpy as np
from matplotlib.patches import Patch
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
class plot_stuff():
    def __init__(self) -> None:
        self.w = None
        

    def plot_regions(self, model, X, y):
        
        x0 = X[X.columns[0]]
        x1 = X[X.columns[1]]
        qual_features = X.columns[2:]
        
        fig, axarr = plt.subplots(1, len(qual_features), figsize = (7, 3))

        # create a grid
        grid_x = np.linspace(x0.min(),x0.max(),501)
        grid_y = np.linspace(x1.min(),x1.max(),501)
        xx, yy = np.meshgrid(grid_x, grid_y)
        
        XX = xx.ravel()
        YY = yy.ravel()

        for i in range(len(qual_features)):
            XY = pd.DataFrame({
                X.columns[0] : XX,
                X.columns[1] : YY
            })

            for j in qual_features:
                XY[j] = 0

            XY[qual_features[i]] = 1

            p = model.predict(XY)
            print(f"p{p}")
            print(type(p))
            p = p.to_numpy()
            p = p.reshape(xx.shape)
            
            
            # use contour plot to visualize the predictions
            # axarr[i].contourf(xx, yy, p, cmap = "jet", alpha = 0.2, vmin = 0, vmax = 2)
            plt.contourf(xx, yy, p, cmap = "jet", alpha = 0.2, vmin = 0, vmax = 2)
            
            ix = X[qual_features[i]] == 1
            # plot the data
            # axarr[i].scatter(x0[ix], x1[ix], c = y[ix], cmap = "jet", vmin = 0, vmax = 2)
            plt.scatter(x0[ix], x1[ix], c = y[ix], cmap = "jet", vmin = 0, vmax = 2)
            
            # axarr[i].set(xlabel = X.columns[0], 
            #         ylabel  = X.columns[1])
            
            # patches = []
            # for color, spec in zip(["red", "green", "blue"], ["Adelie", "Chinstrap", "Gentoo"]):
            #     patches.append(Patch(color = color, label = spec))

            # plt.legend(title = "Species", handles = patches, loc = "best")
            
            plt.tight_layout()

    # from internet
    # decision surface for logistic regression on a binary classification dataset
    def draw(self, X, y, yhat):
        # generate dataset
        # X, y = make_blobs(n_samples=1000, centers=2, n_features=2, random_state=1, cluster_std=3)
        # define bounds of the domain
        min1, max1 = X[:, 0].min()-1, X[:, 0].max()+1
        min2, max2 = X[:, 1].min()-1, X[:, 1].max()+1
        # define the x and y scale
        x1grid = np.arange(min1, max1, 0.1)
        x2grid = np.arange(min2, max2, 0.1)
        # create all of the lines and rows of the grid
        xx, yy = np.meshgrid(x1grid, x2grid)
        # flatten each grid to a vector
        r1, r2 = xx.flatten(), yy.flatten()
        r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
        # horizontal stack vectors to create x1,x2 input for the model
        grid = np.hstack((r1,r2))
        # define the model
        # model = LogisticRegression()
        # fit the model
        # model.fit(X, y)
        # make predictions for the grid
        # yhat = model.predict(grid)
        # reshape the predictions back into a grid
        zz = yhat.reshape(xx.shape)
        # plot the grid of x, y and z values as a surface
        plt.contourf(xx, yy, zz, cmap='Paired')
        # create scatter plot for samples from each class
        for class_value in range(2):
            # get row indexes for samples with this class
            row_ix = np.where(y == class_value)
            # create scatter of these samples
            plt.scatter(X[row_ix, 0], X[row_ix, 1], cmap='Paired')




                    

                


            


    