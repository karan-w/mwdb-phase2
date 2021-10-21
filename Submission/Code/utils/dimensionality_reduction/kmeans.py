from sklearn import cluster
from utils.feature_vector import FeatureVector
import numpy as np

class KMeans:
    def __init__(self) -> None:
        pass

    def compute_centroids(self, dataset_feature_vector, k):
        kmeans = cluster.KMeans(init="k-means++", n_clusters=k)
        kmeans.fit(dataset_feature_vector)
        centroids = kmeans.cluster_centers_ # k * m
        return centroids

    def compute_reduced_dataset_feature_vector(self, centroids, dataset_feature_vector):
        reduced_dataset_feature_vector = []
        for centroid_index in range(centroids.shape[0]):
            centroid = centroids[centroid_index,:]
            distance_centroid = np.sum((dataset_feature_vector-centroid)**2,axis=1)
            reduced_dataset_feature_vector.append(distance_centroid)

        reduced_dataset_feature_vector = np.array(reduced_dataset_feature_vector).T
        return reduced_dataset_feature_vector

    def compute(self, images, k):
        dataset_feature_vector = FeatureVector().create_dataset_feature_vector(images)
        centroids = self.compute_centroids(dataset_feature_vector, k)
        reduced_dataset_feature_vector = self.compute_reduced_dataset_feature_vector(centroids, dataset_feature_vector)
        images = FeatureVector().assign_images_reduced_feature_vector(images, reduced_dataset_feature_vector)
        return images