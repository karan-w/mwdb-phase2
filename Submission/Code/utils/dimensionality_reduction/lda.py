from utils.feature_vector import FeatureVector
from sklearn import decomposition

class LatentDirichletAllocation:
    def __init__(self) -> None:
        pass

    def compute_LDA(self, dataset_feature_vector, k):
        # All the intermediate computations will be stored in the attributes dictionary 
        # so that it can be stored in the output file in the end.     
        attributes = {}
        attributes['dataset_feature_vector'] = dataset_feature_vector
        latent_dirichlet_allocation = decomposition.LatentDirichletAllocation(n_components=k)
        latent_dirichlet_allocation.fit(dataset_feature_vector)
        reduced_dataset_feature_vector = latent_dirichlet_allocation.transform(dataset_feature_vector)
        attributes['reduced_dataset_feature_vector'] = reduced_dataset_feature_vector
        return reduced_dataset_feature_vector, attributes # 400 * k

    def compute_subject_LDA(self, subjects_similarity_matrix, k):
        subjects_latent_semantics, attributes = self.compute_LDA(subjects_similarity_matrix, k)
        return subjects_latent_semantics, attributes

    def compute_type_LDA(self, types_similarity_matrix, k):
        types_latent_semantics, attributes = self.compute_LDA(types_similarity_matrix, k)
        return types_latent_semantics, attributes

    def compute(self, images, k):
        dataset_feature_vector = FeatureVector().create_dataset_feature_vector(images)
        reduced_dataset_feature_vector, attributes = self.compute_LDA(dataset_feature_vector, k)
        images = FeatureVector().assign_images_reduced_feature_vector(images, reduced_dataset_feature_vector)
        return images, attributes