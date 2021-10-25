import argparse
import logging
import json
from sklearn.decomposition import LatentDirichletAllocation
from numpy import dot
from scipy.linalg import svd
from sklearn.datasets import fetch_olivetti_faces
import sklearn
from matplotlib import image
import matplotlib.pyplot as plt
from skimage import feature, exposure, color
import numpy as np
import os
from re import sub
import sys

sys.path.append(".")

from utils.output import Output
from utils.subject import Subject
from utils.dimensionality_reduction.kmeans import KMeans
from utils.dimensionality_reduction.lda import LatentDirichletAllocation
from utils.dimensionality_reduction.svd import SingularValueDecomposition
from utils.dimensionality_reduction.pca import PrincipalComponentAnalysis
from utils.feature_models.hog import HistogramOfGradients
from utils.feature_models.elbp import ExtendedLocalBinaryPattern
from utils.feature_models.cm import ColorMoments
from utils.feature_vector import FeatureVector
from utils.image_reader import ImageReader
from sklearn.cluster import KMeans
from sklearn import preprocessing
from scipy.spatial import distance

COLOR_MOMENTS = 'CM'
EXTENDED_LBP = 'ELBP'
HISTOGRAM_OF_GRADIENTS = 'HOG'

PRINCIPAL_COMPONENT_ANALYSIS = 'PCA'
SINGULAR_VALUE_DECOMPOSITION = 'SVD'
LATENT_DIRICHLET_ALLOCATION = 'LDA'
KMEANS = 'kmeans'

logger = logging.getLogger(__name__)
logging.basicConfig(filename="logs/logs.log", filemode="w", level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S')


class Task3:
    def __init__(self):
        pass

    def setup_args_parser(self):
        parser = argparse.ArgumentParser()

        parser.add_argument(
            '--model', type=str, choices=[COLOR_MOMENTS, EXTENDED_LBP, HISTOGRAM_OF_GRADIENTS], required=True)
        parser.add_argument('--k', type=int, required=True)
        parser.add_argument('--dimensionality_reduction_technique', type=str, choices=[
                            PRINCIPAL_COMPONENT_ANALYSIS, SINGULAR_VALUE_DECOMPOSITION, LATENT_DIRICHLET_ALLOCATION, KMEANS], required=True)
        parser.add_argument('--images_folder_path', type=str, required=True)
        parser.add_argument('--output_folder_path', type=str, required=True)

        return parser

    def log_args(self, args):
        logger.debug("Received the following arguments.")
        logger.debug(f'model - {args.model}')
        logger.debug(f'x - {args.x}')
        logger.debug(f'k - {args.k}')
        logger.debug(
            f'dimensionality_reduction_technique - {args.dimensionality_reduction_technique}')
        logger.debug(f'images_folder_path - {args.images_folder_path}')
        logger.debug(f'output_folder_path - {args.output_folder_path}')

    def compute_feature_vectors(self, feature_model, images):
        if feature_model == COLOR_MOMENTS:
            return ColorMoments().compute(images)
        elif feature_model == EXTENDED_LBP:
            return ExtendedLocalBinaryPattern().compute(images)
        elif feature_model == HISTOGRAM_OF_GRADIENTS:
            return HistogramOfGradients().compute(images)
        else:
            raise Exception(f"Unknown feature model - {feature_model}")

    def get_distinct_image_types(self, images):
        distinct_image_types = set()
        for image in images:
            distinct_image_types.add(image.image_type)
        return sorted(list(distinct_image_types))

    def assign_images_to_types(self, images):
        image_types = self.get_distinct_image_types(images)
        types = []

        # Group images for each distinct subject_id
        for image_type in image_types:
            type_images = []
            for image in images:
                if(image.image_type == image_type):
                    type_images.append(image)
            type = Subject(type_images)
            type.create_type_feature_vector(type.images)
            types.append(type)

        # Return all the distinct subjects as a list
        return types

    # computing subject-subject similarity matrix
    def compute_type_similarity_matrix(self, types):
        type_feature_vector = FeatureVector().create_types_feature_vector(types)
        type_feature_vector_t = np.transpose(type_feature_vector)
        type_dist_matrix = [[0 for x in range(12)] for y in range(12)]

        # instead of multiplying D with D_t we use distance function to calculate distance between corresponsding values
        for i in range(len(type_feature_vector)):
            for j in range(len(type_feature_vector_t[0])):
                for k in range(len(type_feature_vector_t)):
                    type_dist_matrix[i][j] += distance.cityblock(
                        type_feature_vector[i][k], type_feature_vector_t[k][j])

        type_dist_matrix = np.array(type_dist_matrix)

        # reshaping matrix to convert it to 1d array and then normalizing it
        type_dist_matrix = type_dist_matrix.reshape(
            1, len(type_dist_matrix[0])*len(type_dist_matrix[0]))

        # we normazlize the distances
        type_dist_matrix = preprocessing.normalize(
            type_dist_matrix.reshape(1, -1), axis=1, norm='max')

        # reshaping back to 40 x 40 matrix
        type_dist_matrix = type_dist_matrix.reshape(12, 12)

        # using 1-d_norm to calculate actual similairty
        type_similarity_matrix = [[0 for x in range(12)] for y in range(12)]
        for i in range(len(type_dist_matrix[0])):
            for j in range(len(type_dist_matrix[1])):
                type_similarity_matrix[i][j] = 1 - type_dist_matrix[i][j]

        # convert to numpy array
        type_similarity_matrix = np.array(type_similarity_matrix)

        return type_similarity_matrix

    def reduce_dimensions(self, dimensionality_reduction_technique, type_similarity_matrix, k):
        if dimensionality_reduction_technique == PRINCIPAL_COMPONENT_ANALYSIS:
            return PrincipalComponentAnalysis().compute_type_PCA(type_similarity_matrix, k)
        elif dimensionality_reduction_technique == SINGULAR_VALUE_DECOMPOSITION:
            return SingularValueDecomposition().compute_type_SVD(type_similarity_matrix, k)
        elif dimensionality_reduction_technique == LATENT_DIRICHLET_ALLOCATION:
            return LatentDirichletAllocation().compute_type_LDA(type_similarity_matrix, k)
        elif dimensionality_reduction_technique == KMEANS:
            return KMeans().compute_type_KMeans(type_similarity_matrix, k)
        else:
            raise Exception(
                f"Unknown dimensionality reduction technique - {dimensionality_reduction_technique}")

    def build_output(self, args, images, drt_attributes, types, type_weight_matrix, type_similarity_matrix):
        sorted_type_weight_matrix = []
        for i in range(len(type_weight_matrix[0])):
            type_weight_pairs = dict.fromkeys(
                ['Latent Semantic', 'Types', 'Weights'])
            types = []
            weights = []
            for j in range(len(type_weight_matrix)):
                types.append(str(j))
                weights.append(type_weight_matrix[j][i])
            type_weight_pairs['Latent Semantic'] = i
            type_weight_pairs['Weights'] = [
                x for x, _ in sorted(zip(weights, types), reverse=True)]
            type_weight_pairs['Types'] = [
                x for _, x in sorted(zip(weights, types), reverse=True)]
            sorted_type_weight_matrix.append(type_weight_pairs)

        # 2. Prepare dictionary that should be JSONfied to store in JSON file
        output = {
            'type_weight_matrix': sorted_type_weight_matrix,
            'type-type-similarity-matrix': type_similarity_matrix.real.tolist(),
        }
        return output

    def save_output(self, output, output_folder_path):
        OUTPUT_FILE_NAME = 'output.json'
        timestamp_folder_path = Output().create_timestamp_folder(output_folder_path)
        output_json_path = os.path.join(
            timestamp_folder_path, OUTPUT_FILE_NAME)
        Output().save_dict_as_json_file(output, output_json_path)


def main():
    task = Task3()
    parser = task.setup_args_parser()

    # model, x, k, dimensionality_reduction_technique, images_folder_path, output_folder_path
    args = parser.parse_args()
    # task.log_args(args)

    image_reader = ImageReader()

    images = image_reader.get_all_images_in_folder(args.images_folder_path)

    images = task.compute_feature_vectors(args.model, images)

    types = task.assign_images_to_types(images)

    type_similarity_matrix = task.compute_type_similarity_matrix(types)

    type_weight_matrix, drt_attributes = task.reduce_dimensions(
        args.dimensionality_reduction_technique, type_similarity_matrix, args.k)

    output = task.build_output(
        args, images, drt_attributes, types, type_weight_matrix, type_similarity_matrix)

    task.save_output(output, args.output_folder_path)


if __name__ == "__main__":
    main()
