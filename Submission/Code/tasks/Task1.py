import os
import numpy as np
from skimage import feature, exposure, color
import matplotlib.pyplot as plt
from matplotlib import image
import sklearn
from sklearn.datasets import fetch_olivetti_faces
from scipy.linalg import svd
from numpy import dot
from sklearn.decomposition import LatentDirichletAllocation
from gensim.models import LdaModel
import json
import argparse
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

class Task1:
    def __init__(self):
        pass

    def color_moments(self, image_values):
        squeezed_array = []
        mean_moment = []
        sd_moment = []
        skew_moment = []
        current_window = []
        feature_vector = []

        lower_index = 0
        upper_index = 8
        
        while(lower_index != 64 and upper_index != 72):
            for j in range(0, 8):
                for k in range(lower_index, upper_index):
                    current_window.append(image_values[k].reshape(8, 8)[j])
                squeezed_array = np.squeeze(current_window)
                mean_moment.append(squeezed_array.mean())
                sd_moment.append(np.std(squeezed_array))
                skew_moment.append(3*(np.subtract(squeezed_array.mean(), np.median(squeezed_array)))/np.std(squeezed_array))
                current_window.clear()
            if lower_index < 64 and upper_index < 72:
                lower_index+=8
                upper_index+=8

        mean_moment = np.reshape(mean_moment, (8, 8))   #reshaping the array
        sd_moment = np.reshape(sd_moment, (8, 8))       #reshaping the array
        skew_moment = np.reshape(skew_moment, (8, 8))   #reshaping the array
        feature_vector.append([mean_moment, sd_moment, skew_moment])
        feature_vector = np.mean(feature_vector[0], axis=0)

        return feature_vector

    def ELBP(self, input_image):
        #input_image = image_df['image_values'][i].reshape(64, 64)
        ELBP_features = feature.local_binary_pattern(input_image, P=8, R=1, method="ror") #"ror" method is for extension of default 
                                                                                            #implementation which is rotation invariant.
        
        return ELBP_features

    def HOG(self, input_image):
        #hog_image = []
        #input_image = image_df['image_values'][i].reshape(64, 64)
        feature_descrip, image = feature.hog(input_image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys', visualize=True)
        HOG_features = feature_descrip
        #hog_image.append(image)

        return HOG_features

    def PCA(self, feature_vector, k):
        fv_meaned = feature_vector - np.mean(feature_vector, axis=0)
        cov_mat = np.cov(fv_meaned, rowvar=False)
        eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)
        
        #sorting eigen values in descending order
        sorted_index = np.argsort(eigen_values)[::-1]

        sorted_eigenvalue = eigen_values[sorted_index]
        sorted_eigenvectors = eigen_vectors[:,sorted_index]

        latent_ev = sorted_eigenvectors[:,0:k]
        X_reduced = np.dot(latent_ev.transpose(),fv_meaned.transpose()).transpose()

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
        ls = lda_modal.transform(feature_vector)
        return ls

    def features(self, feature_model, imageData):
        data_matrix = []
        if feature_model == 'CM':
            for i in imageData:
                feature_vector = self.color_moments(i) # 1 * 3 * 8 * 8 (8 * 8 + 8 * 8 + 8 * 8) 
                feature_vector = np.squeeze(feature_vector) # Reshapes the matrix to 3 * 8 * 8 
                feature_vector = np.transpose(feature_vector) # 8 * 8 * 3 
                feature_vector = color.rgb2gray(feature_vector) # 8 * 8 - only one CM is retained, other two discarded
                data_matrix.append(feature_vector)
            data_matrix = np.mean(data_matrix, axis=0)

        elif feature_model == 'ELBP':
            for i in imageData:
                feature_vector = self.ELBP(i)
                data_matrix.append(feature_vector)
            data_matrix = np.mean(data_matrix, axis=0)

        elif feature_model == 'HOG':
            for i in imageData:
                feature_vector = self.HOG(i)
                feature_vector = feature_vector.reshape(42, 42)
                data_matrix.append(feature_vector)
            data_matrix = np.mean(data_matrix, axis=0)

        return data_matrix

    def kmeans(self, feature_vector, k):
        k_means = KMeans(n_clusters=k)
        print(feature_vector)
        k_means.fit(feature_vector)
        print(k_means.cluster_centers_)


    def dimension_red(self, technique, feature_vector, k):
        if technique == 'PCA':
            latent_semantic = self.PCA(feature_vector, k)

        elif technique == 'SVD':
            latent_semantic = self.SVD(feature_vector, k)

        elif technique == 'LDA':
            latent_semantic = self.LDA(feature_vector, k)
        
        elif technique == 'kmeans':
            latent_semantic = self.kmeans(feature_vector, k)

        return latent_semantic

    # def kmeans(self, )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--x', type=str, required=True)
    parser.add_argument('--k', type=int, required=True)
    parser.add_argument('--dimensionality_reduction_technique', type=str, required=True)
    parser.add_argument('--images_folder', type=str, required=True)
    parser.add_argument('--output_folder', type=str, required=True)

    args = parser.parse_args()

    feature_model = args.model
    image_type = args.x
    k_value = args.k
    reduction_method = args.dimensionality_reduction_technique
    images_folder = args.images_folder
    output_folder = args.output_folder

    subject_weight_matrix = []

    y = 0
    while(y<40):
        output_dict = dict.fromkeys(['Subject', 'Weight'])
        image_data = []
        y+=1
        for i in range(1, 11):
            image_label = 'image-' + image_type + '-' + str(y) + '-' + str(i) + '.png' 
            image_data.append(image.imread(os.path.join(images_folder, image_label)))
        fv = Task1().features(feature_model, image_data)
        ls = Task1().dimension_red(reduction_method, fv, k_value)
        output_dict['Subject'] = y
        output_dict['Weight'] = ls.tolist()
        subject_weight_matrix.append(output_dict)

    output_json = os.path.join(output_folder, 'data.json') # Create folder with timestamp and store in that

    with open(output_json, 'w') as fp:
        for dictionary in subject_weight_matrix:
            json.dump(dictionary, fp, indent=4)

# TODO: Check all dimensionality reduction techniques
# TODO: Generalize code 
# TODO: Create output folder with timestamp and store in that
