import sys
import numpy as np

sys.path.append(".")

import os

import logging
import argparse
import json
from scipy.spatial.distance import cityblock
from scipy.spatial.distance import euclidean

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

  def compute_feature_vectors(self, feature_model, images):
    if feature_model == COLOR_MOMENTS:
      return ColorMoments().compute(images)
    elif feature_model == EXTENDED_LBP:
      return ExtendedLocalBinaryPattern().compute(images)
    elif feature_model == HISTOGRAM_OF_GRADIENTS:
      return HistogramOfGradients().compute(images)
    else:
      raise Exception(f"Unknown feature model - {feature_model}")

  def compute_query_feature(self, feature_model, image):
    if feature_model == COLOR_MOMENTS:
      return ColorMoments().get_color_moments_fd(image.matrix)
    elif feature_model == EXTENDED_LBP:
      return ExtendedLocalBinaryPattern().get_elbp_fd(image.matrix)
    elif feature_model == HISTOGRAM_OF_GRADIENTS:
      image.feature_vector = HistogramOfGradients().get_hog_fd(image.matrix)
      return image
    else:
      raise Exception(f"Unknown feature model - {feature_model}")

  def read_latent_semantics(self, latent_semantics_file):
    with open(latent_semantics_file, 'r') as f:
      latent_semantics = json.load(f)

    return latent_semantics['args'], latent_semantics['drt_attributes']

  def reduce_dimensions(self, dimensionality_reduction_technique, images, reproject_array):
    if dimensionality_reduction_technique == PRINCIPAL_COMPONENT_ANALYSIS:
      return PrincipalComponentAnalysis().compute_reprojection(images, reproject_array)
    elif dimensionality_reduction_technique == SINGULAR_VALUE_DECOMPOSITION:
      return SingularValueDecomposition().compute(images, reproject_array)
    elif dimensionality_reduction_technique == LATENT_DIRICHLET_ALLOCATION:
      return LatentDirichletAllocation().compute_reprojection(images, reproject_array)
    elif dimensionality_reduction_technique == KMEANS:
      return KMeans().compute_reprojection(images, reproject_array)
    else:
      raise Exception(f"Unknown dimensionality reduction technique - {dimensionality_reduction_technique}")

  def similarity(self, query_image, dataset):
    for image in dataset:
      image.similarity = euclidean(image.reduced_feature_vector, query_image[0].reduced_feature_vector)
    return dataset

if __name__ == "__main__":
  task = Task5()
  parser = task.setup_args_parser()

  # query_image, latent_semantics_file, n, images_folder_path, output_folder_path
  args = parser.parse_args()
  task.log_args(args)

  image_reader = ImageReader()

  query_image = image_reader.get_query_image(args.query_image)
  dataset = image_reader.get_all_images_in_folder(args.images_folder_path)

  metadata, attributes = task.read_latent_semantics(args.latent_semantics_file)

  feature_model = metadata['model']
  dr_technique = metadata['dimensionality_reduction_technique']

  query_image = task.compute_query_feature(feature_model, query_image)
  dataset = task.compute_feature_vectors(feature_model, dataset)

  reproject_matrix = None
  if dr_technique == PRINCIPAL_COMPONENT_ANALYSIS:
    reproject_matrix = np.array(attributes['k_principal_components_eigen_vectors'])
  elif dr_technique == KMEANS:
    reproject_matrix = np.array(attributes['centroids'])
  elif dr_technique == LATENT_DIRICHLET_ALLOCATION:
    reproject_matrix = np.array(attributes['components'])
  # elif feature_model == HISTOGRAM_OF_GRADIENTS:
  #   reproject_matrix = np.array(attributes[''])

  # print(len(reproject_matrix), len(reproject_matrix[0]))

  # if feature_model == KMEANS:
  #   reproject_matrix = np.array(attributes['centroids'])

  query_image = task.reduce_dimensions(dr_technique, [query_image], reproject_matrix)

  dataset = task.reduce_dimensions(dr_technique, dataset, reproject_matrix)

  dataset = task.similarity(query_image, dataset)

  dataset.sort(key=lambda d: d.similarity) 

  print([[dataset[i].filename, dataset[i].similarity] for i in range(args.n)])


