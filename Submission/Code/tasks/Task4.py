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
# from gensim.models import LdaModel
import json
from scipy.spatial import distance
from sklearn import preprocessing
# from Submission.Code.tasks.CommonMethods import CommonMethods


class Task4:
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
                feature_vector = self.color_moments(i)
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

    def dimension_red(self, technique, feature_vector, k):
        if technique == 'PCA':
            latent_semantic = self.PCA(feature_vector, k)

        elif technique == 'SVD':
            latent_semantic = self.SVD(feature_vector, k)

        elif technique == 'LDA':
            latent_semantic = self.LDA(feature_vector, k)

        return latent_semantic

if __name__ == "__main__":
    #parser = argparse.ArgumentParser()

    feature_model = 'ELBP'#str(input('Choose the feature model: '))
    k_value = 10#int(input('Enter the value of k: '))
    reduction_method = 'PCA'#str(input('Choose the dimensionality reduction technique: '))

    #detail images not present
    image_types = ['cc', 'con', 'emboss', 'jitter', 'neg', 'noise01', 'noise02', 'original', 'poster', 'rot', 'smooth', 'stipple']
    subject_images = []

    y = 0
    while(y<40):
        image_data = []
        y+=1
        for image_type in image_types:
            for i in range(1, 10):
                image_label = 'image-' + image_type + '-' + str(y) + '-' + str(i) + '.png'
                image_data.append(image.imread('/content/drive/MyDrive/MWDB_Phase2Project/sample_images/' + image_label))
        fv = Task4().features(feature_model, image_data)
        fv = fv.reshape(np.multiply(np.shape(fv[0]), np.shape(fv[0])))
        subject_images.append(fv)

    #we will use slide 152 of vectors pdf on piazza to create subject-subject similarity matrix
    #i.e using DxD_t
    subject_images_t = np.transpose(subject_images)
    dist_dist_matrix = [[0 for x in range(40)] for y in range(40)]

    #instead of multiplying D with D_t we use distance function to calculate distance between corresponsding values
    for i in range(len(subject_images)):
        for j in range(len(subject_images_t[0])):
            for k in range(len(subject_images_t)):
                dist_dist_matrix[i][j] += distance.cityblock(subject_images[i][k], subject_images_t[k][j])
    
    dist_dist_matrix = np.array(dist_dist_matrix)

    #we normazlize the distances
    dist_dist_matrix = preprocessing.normalize(dist_dist_matrix)

    #using 1-d_norm to calculate actual similairty
    sim_sim_matrix = [[0 for x in range(40)] for y in range(40)]
    for i in range(len(dist_dist_matrix[0])):
        for j in range(len(dist_dist_matrix[1])):
            sim_sim_matrix[i][j] = 1 - dist_dist_matrix[i][j]

    #convert to numpy array
    sim_sim_matrix = np.array(sim_sim_matrix)

    ls = Task4().dimension_red(reduction_method, sim_sim_matrix, k_value)

    output = []
    for i in range(len(ls[0])):
        subject_weight_pairs = dict.fromkeys(['Latent Semantic', 'Subjects', 'Weights'])
        subjects = []
        weights = []
        for j in range(len(ls)):
            subjects.append(str(j))
            weights.append(ls[j][i])
        subject_weight_pairs['Latent Semantic'] = i
        subject_weight_pairs['Subjects'] = [x for x,_ in sorted(zip(subjects,weights), reverse=True)]
        subject_weight_pairs['Weights'] = [x for _,x in sorted(zip(subjects,weights), reverse=True)]
        output.append(subject_weight_pairs)

    with open('task4-output.json', 'w') as fp:
        for dictionary in output:
            json.dump(dictionary, fp, indent=4)

    # parser.add_argument('--model', type=str, required=True)
    # parser.add_argument('--x', type=str, required=True)
    # parser.add_argument('--k', type=int, required=True)
    # parser.add_argument('--dimensionality_reduction_technique', type=str, required=True)

    #args = parser.parse_args()