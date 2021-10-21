import numpy as np

from utils.feature_vector import FeatureVector

class PrincipalComponentAnalysis:
    def __init__(self) -> None:
        pass

    def standardize_dataset_feature_vector(self, dataset_feature_vector):
        mean_dataset_feature_vector = np.mean(dataset_feature_vector, axis=0)
        std_dataset_feature_vector = np.std(dataset_feature_vector, axis=0)
        return ((dataset_feature_vector - mean_dataset_feature_vector)/std_dataset_feature_vector)

    def compute_PCA(self, dataset_feature_vector, k):
        standardized_dataset_feature_vector = self.standardize_dataset_feature_vector(dataset_feature_vector)
        covariance_matrix = np.cov(dataset_feature_vector, rowvar=False) # m x m

        '''
        eigen_vectors are normalized such that the column v[:,i] 
        is the eigenvector corresponding to the eigenvalue w[i]
        '''
        eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)

        #sorting eigen values in descending order
        sorted_index = np.argsort(eigen_values)[::-1]

        sorted_eigenvalue = eigen_values[sorted_index]
        sorted_eigenvectors = eigen_vectors[:,sorted_index]

        k_principal_components_eigen_vectors = sorted_eigenvectors[:,0:k]

        reduced_dataset_feature_vector = np.dot(dataset_feature_vector, k_principal_components_eigen_vectors) # (400, 1764) * (1764, 2)
        return reduced_dataset_feature_vector

    

    def compute(self, images, k):
        dataset_feature_vector = FeatureVector().create_dataset_feature_vector(images)
        reduced_dataset_feature_vector = self.compute_PCA(dataset_feature_vector, k)
        images = FeatureVector().assign_images_reduced_feature_vector(images, reduced_dataset_feature_vector)
        return images

        # TODO: Assign reduced feature descriptor to each image

