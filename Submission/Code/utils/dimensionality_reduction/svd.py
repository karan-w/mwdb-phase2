import numpy as np
from scipy.linalg import svd

from utils.feature_vector import FeatureVector

class SingularValueDecomposition:
    def __init__(self) -> None:
        pass

    def compute_SVD(self, dataset_feature_vector, k):
        left_factor_matrix, core_matrix, right_factor_matrix = svd(dataset_feature_vector)

        # Reduce the core matrix 

        reduced_core_matrix = np.zeros((dataset_feature_vector.shape[0], dataset_feature_vector.shape[1])) # n * m

        #populating sigma with nxn diagonal matrix
        reduced_core_matrix[:dataset_feature_vector.shape[0], :dataset_feature_vector.shape[0]] = np.diag(core_matrix)  

        k_latent = k
        reduced_core_matrix = reduced_core_matrix[:, :k_latent]
        right_factor_matrix = right_factor_matrix[:k_latent, :]
        reduced_dataset_feature_vector = left_factor_matrix.dot(reduced_core_matrix)
        return reduced_dataset_feature_vector

    def compute(self, images, k):
        dataset_feature_vector = FeatureVector().create_dataset_feature_vector(images)
        reduced_dataset_feature_vector = self.compute_SVD(dataset_feature_vector, k)
        images = FeatureVector().assign_images_reduced_feature_vector(images, reduced_dataset_feature_vector)
        return images
        
        
    # 1. Left Factor Matrix 
    # 2. Core matrix
    # 3. Right factor matrix