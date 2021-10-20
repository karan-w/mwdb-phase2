import numpy as np

class PrincipalComponentAnalysis:
    def __init__(self) -> None:
        pass

    def create_dataset_feature_vector(self, images):
        images = sorted(images, key=lambda image: (image.subject_id, image.image_id))
        dataset_feature_vector = []
        for image in images:
            dataset_feature_vector.append(image.feature_vector)
        dataset_feature_vector = np.stack(dataset_feature_vector, axis=0) # 400 x fv_size
        return dataset_feature_vector

    def standardize_dataset_feature_vector(self, dataset_feature_vector):
        mean_dataset_feature_vector = np.mean(dataset_feature_vector, axis=0)
        std_dataset_feature_vector = np.std(dataset_feature_vector, axis=0)
        return ((dataset_feature_vector - mean_dataset_feature_vector)/std_dataset_feature_vector)

    def compute(self, images, k):
        dataset_feature_vector = self.create_dataset_feature_vector(images)
        standardized_dataset_feature_vector = self.standardize_dataset_feature_vector(dataset_feature_vector)
        covariance_matrix = np.cov(standardized_dataset_feature_vector, rowvar=False) # m x m
        eigen_values, eigen_vectors = np.linalg.eigh(covariance_matrix)

        #sorting eigen values in descending order
        sorted_index = np.argsort(eigen_values)[::-1]

        sorted_eigenvalue = eigen_values[sorted_index]
        sorted_eigenvectors = eigen_vectors[:,sorted_index]

        k_principal_components_eigen_vectors = sorted_eigenvectors[:,0:k]
        reduced_dataset_feature_vector = np.dot(standardized_dataset_feature_vector, k_principal_components_eigen_vectors) # (400, 1764) * (1764, 2)

        return reduced_dataset_feature_vector