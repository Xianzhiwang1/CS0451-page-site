from sklearn.preprocessing import LabelEncoder
from matplotlib.patches import Patch
from matplotlib import pyplot as plt
plt.rcParams["figure.figsize"] = (4,4)
import numpy as np
np.random.seed(42)
from itertools import combinations
import pandas as pd
le = LabelEncoder()
# train_url = "https://raw.githubusercontent.com/middlebury-csci-0451/CSCI-0451/main/data/palmer-penguins/train.csv"
# train = pd.read_csv(train_url)

class PG():
    def __init__(self, df_col):
        self.le = le.fit(df_col)

    def prepare_data(self, df):
        df = df.drop(["studyName", "Sample Number", "Individual ID", "Date Egg", "Comments", "Region"], axis = 1)
        df = df[df["Sex"] != "."]
        df = df.dropna()
        y = self.le.transform(df["Species"])
        df = df.drop(["Species"], axis = 1)
        df = pd.get_dummies(df)
        return df, y

    def select_combin(self, df, all_qual_cols, all_quant_cols):
        for qual in all_qual_cols: 
            qual_cols = [col for col in df.columns if qual in col ]
            for pair in combinations(all_quant_cols, 2):
                cols = qual_cols + list(pair) 
                print(cols)
                # you could train models and score them here, keeping the list of 
                # columns for the model that has the best score. 
                # 


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
            p = p.reshape(xx.shape)
            
            
            # use contour plot to visualize the predictions
            axarr[i].contourf(xx, yy, p, cmap = "jet", alpha = 0.2, vmin = 0, vmax = 2)
            
            ix = X[qual_features[i]] == 1
            # plot the data
            axarr[i].scatter(x0[ix], x1[ix], c = y[ix], cmap = "jet", vmin = 0, vmax = 2)
            
            axarr[i].set(xlabel = X.columns[0], 
                    ylabel  = X.columns[1])
            
            patches = []
            for color, spec in zip(["red", "green", "blue"], ["Adelie", "Chinstrap", "Gentoo"]):
                patches.append(Patch(color = color, label = spec))

            plt.legend(title = "Species", handles = patches, loc = "best")
            
            plt.tight_layout()