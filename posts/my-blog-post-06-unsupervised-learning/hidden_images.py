
from matplotlib import pyplot as plt
import numpy as np
np.random.seed(42)


class svd:
    def __init__(self) -> None:
        self.sigma = None
        
    def reconstruct(self, img: np.array, k: int):
        U, self.sigma, V = np.linalg.svd(img)
        # create the Sigma matrix in the SVD
        Sigma = np.zeros_like(img,dtype=float) # matrix of zeros of same shape as matrix img 
        Sigma[:min(img.shape),:min(img.shape)] = np.diag(self.sigma)        # singular values on the main diagonal

        # pick first k col of U
        # pick top k singular values of Sigma
        # pick first k row of V
        if (k <= img.shape[0]) and (k <= img.shape[1]):
            U_ = U[:,:k]
            Sigma_ = Sigma[:k, :k]
            V_ = V[:k, :]
            A_ = U_ @ Sigma_ @ V_
            return A_



    def experiment(self, img):
        fig, axarr = plt.subplots(3, 2, figsize = (4, 4))
        n_pixel_img = (img.shape[0] * img.shape[1])
        i = 3 
        for ax in axarr.ravel():
            A_ = self.reconstruct(img, i)
            n_pixel_A_ = (A_.shape[0] * A_.shape[1])
            storage = ( n_pixel_A_ / n_pixel_img ) * 100
            ax.imshow(A_, cmap = "Greys")
            ax.axis("off")
            ax.set(title = f"{i} components, storage = {storage} % ")
            i += 5 

        plt.tight_layout()

    def compare_images(self, A: np.array, A_: np.array) -> None:

        fig, axarr = plt.subplots(1, 2, figsize = (7, 3))

        axarr[0].imshow(A, cmap = "Greys")
        axarr[0].axis("off")
        axarr[0].set(title = "original image")

        axarr[1].imshow(A_, cmap = "Greys")
        axarr[1].axis("off")
        axarr[1].set(title = "reconstructed image")

        
            
