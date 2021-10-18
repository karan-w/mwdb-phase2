import numpy as np
from skimage import feature
from matplotlib import image
from sklearn.decomposition import LatentDirichletAllocation
from glob import glob
import pandas as pd
import scipy
import os
from scipy.linalg import svd

class Task2:
    def __init__(self):
        pass

    def color_moments(self, X):
        moment_m = []
        moment_v = []
        moment_s = []

        # Looping through the 64x64 image to calculate moments for 8x8 blocks
        for i in range(0, len(X), 8):
            for j in range(0, len(X), 8):
                block = X[i:i + 8, j:j + 8]
                moment_m.append(np.mean(block))
                moment_v.append(np.var(block))
                moment_s.append(scipy.stats.skew(block, axis=None))

        # Combining moments to make a (3, 8, 8) size feature vector
        moment = [moment_m, moment_v, moment_s]
        moment = np.array(moment)
        return moment.reshape(3*8*8)

    def ELBP(self, X):
        radius = 2
        n_points = 4 * radius
        lbp = feature.local_binary_pattern(X, n_points, radius, method="ror")
        (hist, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, n_points + 3),
                                 range=(0, n_points + 2))
        hist.astype("float")
        hist = hist / hist.sum()
        return hist

    def HOG(self, X):
        out, hog = feature.hog(X, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys', visualize=True)
        return out

    def PCA(self, feature_vector, k):
        fv_meaned = feature_vector - np.mean(feature_vector, axis=0)
        cov_mat = np.cov(fv_meaned, rowvar=False)
        eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)
        
        #sorting eigen values in descending order
        sorted_index = np.argsort(eigen_values)[::-1]

        sorted_eigenvalue = eigen_values[sorted_index]
        sorted_eigenvectors = eigen_vectors[:, sorted_index]

        latent_ev = sorted_eigenvectors[:, 0:k]
        X_reduced = np.dot(latent_ev.transpose(), fv_meaned.transpose()).transpose()

        return X_reduced

    def SVD(self, feature_vector, k):
        U, s, V_t = svd(feature_vector)

        #creating mxn sigma matrix
        Sigma = np.zeros((feature_vector.shape[0], feature_vector.shape[1]))

        #populating sigma with nxn diagonal matrix
        Sigma[:feature_vector.shape[0], :feature_vector.shape[0]] = np.diag(s)

        k_latent = k
        Sigma = Sigma[:, :k_latent]
        V_t = V_t[:k_latent, :]
        transformed_matrix = U.dot(Sigma)

        return transformed_matrix

    def LDA(self, feature_vector, k):
        lda_modal = LatentDirichletAllocation(n_components=k)
        lda_modal.fit(feature_vector)
        lda = lda_modal.transform(feature_vector)
        return lda

    def features(self, feature_model, imageData):
        feature_vector = None
        if feature_model == 'CM':
            feature_vector = self.color_moments(imageData)

        elif feature_model == 'ELBP':
            feature_vector = self.ELBP(i)

        elif feature_model == 'HOG':
            feature_vector = self.HOG(imageData)

        return feature_vector

    def dimension_red(self, technique, feature_vector, k):
        if technique == 'PCA':
            latent_semantic = self.PCA(feature_vector, k)

        elif technique == 'SVD':
            latent_semantic = self.SVD(feature_vector, k)

        elif technique == 'LDA':
            latent_semantic = self.LDA(feature_vector, k)

        return latent_semantic


if __name__ == "__main__":

    os.chdir('C:/Users/sshah96/Desktop/ASU/CSE515/Project Phase 2/phase2_data/all/')   

    feature_model = str(input('Choose the feature model: '))
    subject_id = str(input('Choose Subject ID: '))
    k_value = int(input('Enter the value of k: '))
    reduction_method = str(input('Choose the dimensionality reduction technique: '))

    type_weight_matrix = []

    regex = '*-' + subject_id + '-*.png'

    files = glob(regex)

    files.sort()

    types_of_image = set()

    for file in files:
        start = file.find('-') + 1
        end = file.find('-', start)

        types_of_image.add(file[start:end])

    types_of_image = list(types_of_image)
    types_of_image.sort()
    types_of_image = np.array(types_of_image)
    types_of_image = np.reshape(types_of_image, (len(types_of_image), 1))

    fv = []
    for file in files:
        image_data = image.imread(file)
        fv.append(Task2().features(feature_model, image_data))

    ls = Task2().dimension_red(reduction_method, fv, k_value)

    type_weight_matrix = []
    dis = int(len(ls) / (len(types_of_image)))
    for i in range(0, len(ls), dis):
        type_weight_matrix.append(np.mean(ls[i:i + dis], axis=0))

    type_weight_matrix = np.array(type_weight_matrix)
    types_weight_matrix = type_weight_matrix.T

    latent_semantic_file = pd.DataFrame(data=np.hstack((types_of_image, type_weight_matrix)))
    file_name = feature_model + '_' + subject_id + '_' + str(k_value) + '_' + reduction_method + '.csv'
    latent_semantic_file.to_csv('../' + file_name, sep=',')

    print(latent_semantic_file)
