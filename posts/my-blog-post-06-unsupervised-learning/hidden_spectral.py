
from sklearn.datasets import make_blobs, make_circles
from matplotlib import pyplot as plt
import numpy as np
import numpy.linalg as linalg
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances

class spectral:
    def __init__(self) -> None:
        self.L = None
        self.D = None
        self.X = None

    def cut(self, A, z):
        D = pairwise_distances(z.reshape(-1, 1))
        return (A*D).sum()

    def vol(self, j, A, z):
        return A[z == j,:].sum()

    def normcut(self, A, z):
        return self.cut(A, z) * (1/self.vol(0, A, z) + 1/self.vol(1, A, z))

    def second_laplacian_eigenvector(self, A: np.array):
        self.D = np.diag(A.sum(axis = 0))
        # print(self.D)
        self.L = np.multiply(np.linalg.inv(self.D), (self.D-A))
        # print(self.L)
        evalue, evector = np.linalg.eig(self.L)
        # print("evalue")
        # print(evalue)
        # print("evector")
        # print(evector)
        k = self.L.shape[1] 
        idx = evalue.argsort()[:k][::-1] 
        evalue = evalue[idx]
        evector = evector[:, idx]
        # access eigenvector associated with the second smallest eigenvalue 
        index = evector.shape[1]
        return evector[:, index-2] 



    def spectral_clustering(self, G, n_neighbors = 6) -> np.array:

        return None



    def plot_graph(self, X, A, z = None, ax = None, show_edge_cuts = True):
        G = nx.from_numpy_array(A)
        if z is None:
            nx.draw(G, pos = X, alpha = .4, node_color = "grey", node_size = 20, ax = ax)
        else: 
            if show_edge_cuts:
                colors = ["red" if z[i] != z[j] else "grey" for i, j in G.edges()]
                widths = [2 if z[i] != z[j] else 1 for i, j in G.edges()]
            else:
                colors = "black"
                widths = 1
            
            nx.draw(G, pos = X, alpha = .4, node_color = z, node_size = 20, edge_color = colors, width = widths, ax = ax, cmap=plt.cm.cividis)

        plt.gca().set(xlabel = "Feature 1", ylabel = "Feature 2")





