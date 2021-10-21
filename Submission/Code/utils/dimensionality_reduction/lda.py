from utils.feature_vector import FeatureVector
from sklearn import decomposition

class LatentDirichletAllocation:
    def __init__(self) -> None:
        pass

    def compute_LDA(self, dataset_feature_vector, k):
        latent_dirichlet_allocation = decomposition.LatentDirichletAllocation(n_components=k)
        latent_dirichlet_allocation.fit(dataset_feature_vector)
        reduced_dataset_feature_vector = latent_dirichlet_allocation.transform(dataset_feature_vector)
        return reduced_dataset_feature_vector # 400 * k

    def compute(self, images, k):
        dataset_feature_vector = FeatureVector().create_dataset_feature_vector(images)
        reduced_dataset_feature_vector = self.compute_LDA(dataset_feature_vector, k)
        images = FeatureVector().assign_images_reduced_feature_vector(images, reduced_dataset_feature_vector)
        return images