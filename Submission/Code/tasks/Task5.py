import sys
import numpy as np

sys.path.append(".")

import os

import logging
import argparse
import json
from scipy.spatial.distance import cityblock
from scipy.spatial.distance import euclidean
import cv2
import matplotlib.pyplot as plt
from utils.image_reader import ImageReader
from utils.feature_models.cm import ColorMoments
from utils.feature_models.elbp import ExtendedLocalBinaryPattern
from utils.feature_models.hog import HistogramOfGradients
from utils.dimensionality_reduction.pca import PrincipalComponentAnalysis
from utils.dimensionality_reduction.svd import SingularValueDecomposition
from utils.dimensionality_reduction.lda import LatentDirichletAllocation
from utils.dimensionality_reduction.kmeans import KMeans
from utils.subject import Subject
from utils.feature_vector import FeatureVector
from utils.output import Output

COLOR_MOMENTS = 'CM'
EXTENDED_LBP = 'ELBP'
HISTOGRAM_OF_GRADIENTS = 'HOG'

PRINCIPAL_COMPONENT_ANALYSIS = 'PCA'
SINGULAR_VALUE_DECOMPOSITION = 'SVD'
LATENT_DIRICHLET_ALLOCATION = 'LDA'
KMEANS = 'kmeans'

logger = logging.getLogger(__name__)
logging.basicConfig(filename="logs/logs.log", filemode="w", level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S')

class Task5:
  def __init__(self):
        pass

  def setup_args_parser(self):
    parser = argparse.ArgumentParser()

    parser.add_argument('--query_image', type=str)
    parser.add_argument('--latent_semantics_file', type=str, required=True)
    parser.add_argument('--n', type=int, required=True)
    parser.add_argument('--images_folder_path', type=str, required=True)
    parser.add_argument('--output_folder_path', type=str, required=True)
    
    return parser

  def log_args(self, args):
    logger.debug("Received the following arguments.")
    logger.debug(f'query_image - {args.query_image}')
    logger.debug(f'latent_semantics_file - {args.latent_semantics_file}')
    logger.debug(f'n - {args.n}')
    logger.debug(f'images_folder_path - {args.images_folder_path}')
    logger.debug(f'output_folder_path - {args.output_folder_path}')

  # def compute_feature_vectors(self, feature_model, images):
  #   if feature_model == COLOR_MOMENTS:
  #     return ColorMoments().compute(images)
  #   elif feature_model == EXTENDED_LBP:
  #     return ExtendedLocalBinaryPattern().compute(images)
  #   elif feature_model == HISTOGRAM_OF_GRADIENTS:
  #     return HistogramOfGradients().compute(images)
  #   else:
  #     raise Exception(f"Unknown feature model - {feature_model}")

  def compute_query_feature_vector(self, feature_model, image):
    if feature_model == COLOR_MOMENTS:
      return ColorMoments().get_color_moments_fd(image.matrix)
    elif feature_model == EXTENDED_LBP:
      return ExtendedLocalBinaryPattern().get_elbp_fd(image.matrix)
    elif feature_model == HISTOGRAM_OF_GRADIENTS:
      image.feature_vector = HistogramOfGradients().get_hog_fd(image.matrix)
      return image
    else:
      raise Exception(f"Unknown feature model - {feature_model}")

  def compute_reprojection_matrix(self, drt_technique):
    reproject_matrix = None
    if drt_technique == PRINCIPAL_COMPONENT_ANALYSIS:
      reproject_matrix = np.array(attributes['k_principal_components_eigen_vectors'])
    elif drt_technique == KMEANS:
      reproject_matrix = np.array(attributes['centroids'])
    elif drt_technique == LATENT_DIRICHLET_ALLOCATION:
      reproject_matrix = np.array(attributes['components'])
    elif drt_technique == SINGULAR_VALUE_DECOMPOSITION:
      reproject_matrix = np.array(attributes['right_factor_matrix'])

    return reproject_matrix

  def read_latent_semantics(self, latent_semantics_file):
    with open(latent_semantics_file, 'r') as f:
      latent_semantics = json.load(f)

    return latent_semantics['args'], latent_semantics['drt_attributes'], latent_semantics['images']

  def reduce_dimensions(self, dimensionality_reduction_technique, query_image, reprojection_matrix):
    if dimensionality_reduction_technique == PRINCIPAL_COMPONENT_ANALYSIS:
      return PrincipalComponentAnalysis().compute_reprojection(query_image, reprojection_matrix)
    elif dimensionality_reduction_technique == SINGULAR_VALUE_DECOMPOSITION:
      return SingularValueDecomposition().compute_reprojection(query_image, reprojection_matrix)
    elif dimensionality_reduction_technique == LATENT_DIRICHLET_ALLOCATION:
      return LatentDirichletAllocation().compute_reprojection(query_image, reprojection_matrix)
    elif dimensionality_reduction_technique == KMEANS:
      return KMeans().compute_reprojection(query_image, reprojection_matrix)
    else:
      raise Exception(f"Unknown dimensionality reduction technique - {dimensionality_reduction_technique}")

  def compute_similarity(self, reduced_query_feature_vector, dataset):
    for image in dataset:
      # reduced_shape = np.shape(reduced_query_feature_vector)
      # reduced_image = reduced_query_feature_vector
      # this_shape = np.shape(image['reduced_feature_vector'])
      # dataset_image = image['reduced_feature_vector']
      distance = cityblock(reduced_query_feature_vector, image['reduced_feature_vector'])
      image['similarity'] = 1 / distance
    return dataset

if __name__ == "__main__":
  task = Task5()
  parser = task.setup_args_parser()

  # query_image, latent_semantics_file, n, images_folder_path, output_folder_path
  args = parser.parse_args()
  task.log_args(args)

  image_reader = ImageReader()

  query_image = image_reader.get_query_image(args.query_image)
  # dataset = image_reader.get_all_images_in_folder(args.images_folder_path)

  metadata, attributes, images = task.read_latent_semantics(args.latent_semantics_file)

  feature_model = metadata['model']
  drt_technique = metadata['dimensionality_reduction_technique']

  query_image = task.compute_query_feature_vector(feature_model, query_image)
  # dataset = task.compute_feature_vectors(feature_model, dataset)

  query_image = np.reshape(query_image, (1, -1))
  #collecting the reprojection matrix (1 x m) to reduce (or reproject) query image onto latent space
  reprojection_matrix = task.compute_reprojection_matrix(drt_technique)

  sh = np.shape(reprojection_matrix)
  reduced_query_feature_vector = task.reduce_dimensions(drt_technique, query_image, reprojection_matrix)

  similarity_with_query_image = task.compute_similarity(reduced_query_feature_vector, images)

  sorted_images = sorted(similarity_with_query_image, key=lambda d: d['similarity'], reverse=True) 

  most_n_similar_images = [[sorted_images[i]['filename'], sorted_images[i]['similarity']] for i in range(args.n)]

  for i in range(args.n):
    name = args.images_folder_path + "/" + most_n_similar_images[i][0], cv2.IMREAD_GRAYSCALE
    image = cv2.imread(args.images_folder_path + "/" + most_n_similar_images[i][0], cv2.IMREAD_GRAYSCALE)
    plt.figure()
    plt.imshow(image)
    plt.title(most_n_similar_images[i][0])
    plt.show()
