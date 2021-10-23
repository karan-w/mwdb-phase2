import numpy as np
from scipy.linalg import svd

from utils.feature_vector import FeatureVector

class SingularValueDecomposition:
    def __init__(self) -> None:
        pass

    def compute_SVD(self, dataset_feature_vector, k):
        # All the intermediate computations will be stored in the attributes dictionary 
        # so that it can be stored in the output file in the end.     
        attributes = {}
        left_factor_matrix, core_matrix, right_factor_matrix = svd(dataset_feature_vector)

        # Reduce the core matrix 

        reduced_core_matrix = np.zeros((dataset_feature_vector.shape[0], dataset_feature_vector.shape[1])) # n * m

        #populating sigma with nxn diagonal matrix
        reduced_core_matrix[:dataset_feature_vector.shape[0], :dataset_feature_vector.shape[0]] = np.diag(core_matrix)  

        k_latent = k
        reduced_core_matrix = reduced_core_matrix[:, :k_latent]
        right_factor_matrix = right_factor_matrix[:k_latent, :]
        reduced_dataset_feature_vector = left_factor_matrix.dot(reduced_core_matrix)

        return reduced_dataset_feature_vector, attributes

    def compute_subject_SVD(self, subjects_similarity_matrix, k):
        subjects_latent_semantics, attributes = self.compute_SVD(subjects_similarity_matrix, k)
        return subjects_latent_semantics, attributes

    def compute_type_SVD(self, types_similarity_matrix, k):
        types_latent_semantics, attributes = self.compute_SVD(types_similarity_matrix, k)
        return types_latent_semantics, attributes

    def compute(self, images, k):
        dataset_feature_vector = FeatureVector().create_dataset_feature_vector(images)
        reduced_dataset_feature_vector, attributes = self.compute_SVD(dataset_feature_vector, k)
        images = FeatureVector().assign_images_reduced_feature_vector(images, reduced_dataset_feature_vector)
        return images, attributes

    def compute2(self, object_feature_vector, k):
        reduced_dataset_feature_vector, attributes = self.compute_SVD(object_feature_vector, k)
        return reduced_dataset_feature_vector
        
    # 1. Left Factor Matrix 
    # 2. Core matrix
    # 3. Right factor matrix