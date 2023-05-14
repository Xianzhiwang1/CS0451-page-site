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


class FinalProject:

    def __init__(self) -> None:
        self.qual_cols = None
        self.feature_score_pair = dict()
        self.beta = None
        self.mypredict = None

    def try_to_plot_decision_regioin(self, X_train, y_train, cols):
        clf = SVC(C=100,gamma=0.0001)
        pca = PCA(n_components = 2)
        X_train2 = pca.fit_transform(X_train[cols].to_numpy())
        clf.fit(X_train[cols], y_train.to_numpy())
        plot_decision_regions(X_train[cols].to_numpy(), y_train.to_numpy(), clf=clf, legend=2)

        plt.xlabel(X_train[cols].to_numpy().columns[0], size=14)
        plt.ylabel(X_train[cols].to_numpy().columns[1], size=14)
        plt.title('SVM Decision Region Boundary', size=16)



    def convert_stata(self):
        '''
        We only need to use this function once, and in subsequent analysis, we simply load in the .csv file.
        '''
        Rvss = pd.io.stata.read_stata("./../Rvssian/AG_Corp_RuscorpMasterFile_Cleaned.dta")
        Rvss.to_csv("RvssianCorpMasterFileCleaned.csv")
        Rvss_data = pd.io.stata.read_stata("./AG_Corp_Prod_Database.dta")
        Rvss_data.to_csv("AG_Corp_Prod_DataBase.csv")



    def make_ready_for_regression(self, X, y, cols):
        # X
        X = X[cols]
        X = X.fillna(0)
        X = X.to_numpy()
        # y 
        y = y.fillna(0)
        y = y.to_numpy()
        y = y.reshape(-1,1)
        return X,y

    def create_balanced_data(self, df) -> pd.DataFrame:
        df_inc = df.loc[df['Form'] == 1]
        df_not_inc = df.loc[df['Form'] == 0]
        print(f"df incorporated have {df_inc.shape[0]} many rows")
        df_not_inc = df_not_inc.sample(n=2393, replace=False)
        print(f"after balancing, df not incorporated have {df_not_inc.shape[0]} many rows")
        frames = [df_inc, df_not_inc]
        result = pd.concat(frames)
        result = result.sample(frac=1).reset_index(drop=True)
        return result
        
    def balanced_one_minus_one(self, df) -> pd.DataFrame:
        df_inc = df.loc[df['Form'] == 1]
        df_not_inc = df.loc[df['Form'] == -1]
        print(f"df incorporated have {df_inc.shape[0]} many rows")
        df_not_inc = df_not_inc.sample(n=2393, replace=False)
        print(f"after balancing, df not incorporated have {df_not_inc.shape[0]} many rows")
        frames = [df_inc, df_not_inc]
        result = pd.concat(frames)
        result = result.sample(frac=1).reset_index(drop=True)
        return result

    def plot_regions(self, model, X, y):
        pass

    def prepare_data(self, df):
        y = df['Form']
        X = df.drop(['Form'], axis = 1)
        return df, X, y

    def split_data(self, df):
        train, validate, test = np.split(df.sample(frac=1, random_state=42), [int(.6*len(df)), int(.8*len(df))])
        return train, validate, test  

    def feature_combo(self, all_qual_cols, all_quant_cols, df, y): 
        for qual in all_qual_cols: 
            self.qual_cols = [col for col in df.columns if qual in col ]
            for pair in combinations(all_quant_cols, 2):
                cols = self.qual_cols + list(pair) 
                # you could train models and score them here, keeping the list of 
                # columns for the model that has the best score. 
                LR = LogisticRegression()
                LR.fit(df[cols], y)
                score = LR.score(df[cols], y)
                self.feature_score_pair[tuple(cols)] = score


    def print_confusion_matrix(self, model, X, y):
        my_matr = confusion_matrix(y, model.predict(X), normalize="true")
        fig, ax = plt.subplots(figsize=(4,4))
        ax.imshow(my_matr)
        ax.xaxis.set(ticks=(0,1), ticklabels=('Predicted False', 'Predicted True'))
        ax.yaxis.set(ticks=(0,1), ticklabels=('Actually False', 'Actually True'))
        ax.set_ylim(1.5, -0.5)

        for i in range(2):
            for j in range(2):
                ax.text(j,i, my_matr[i,j].round(3), ha='center', va='center', color='black')

    def encode_features(self, df: pd.DataFrame, var_name: str):
        le = LabelEncoder()
        le.fit(df[var_name])
        var_name_coded = le.transform(df[var_name])
        df_minus = df.drop(columns=var_name, axis = 1)
        return df_minus, var_name_coded 

    def all_the_columns(self):
        cols = ['id', 'FoundingYear', 'Province', 'OntheSide', 'Age', 'TaxedActivity',
            'YEAR', 'PSZLastYear', 'PSZ1908', 'SubindustryCode', 'STCAP', 'Revenue',
            'TotalWorkers', 'TotalPower', 'GrandTotalWorkers', 'RevperWorker',
            'PowerperWorker', 'RevperGrandWorker', 'PowerperGrandWorker',
            'logRevperWorker', 'logPowerperWorker', 'logRevperGrandWorker',
            'logPowerperGrandWorker', 'logRev', 'logWorkers', 'logPower',
            'RegIndGroup', 'RegIndYearGroup', 'ProvIndGroup', 'ProvIndYearGroup',
            'IndYearGroup', 'IndustryFactor', 'ProvinceFactor', 'YearFactor',
            'AKTS', 'PAI', 'factory_id', 'FormNextYear', 'FormNextNextYear',
            'FactoryisCorpin1894', 'FormNextYearin1894', 'FactoryisCorpin1900',
            'FormNextYearin1900', 'FactoryisCorpin1908', 'NEWDEV', 'SHARES',
            'STPRICE', 'BONDS', 'Silk', 'Flax', 'Animal', 'Wool', 'Cotton',
            'MixedMaterials', 'Wood', 'Paper', 'MetalsandMachines', 'Foods',
            'Chemical', 'Mineral']
        return cols

    def poly_LR(self, deg):
        return Pipeline([("poly", PolynomialFeatures(degree=deg)),
                        ("LR", LogisticRegression(penalty="none", max_iter=int(1e3)))])


    def polynomial_degree_validation(self, df, y, max_deg, cv): 
        for deg in range(max_deg):
            plr = self.poly_LR(deg = deg)
            cv_scores = cross_val_score(plr, df, y, cv=cv)
            mean_score = cv_scores.mean()  
            print(f"Polynomial degree = {deg}, score = {mean_score.round(3)}")


    def preparing_data(self, df):
        # le.fit(df["Species"])
        # df = df.drop(["studyName", "Sample Number", "Individual ID", "Date Egg", "Comments", "Region"], axis = 1)
        # df = df[df["Sex"] != "."]
        # df = df.dropna()
        # y = le.transform(df["Species"])
        # df = df.drop(["Species"], axis = 1)
        # df = pd.get_dummies(df)
        # return df, y
        pass




































