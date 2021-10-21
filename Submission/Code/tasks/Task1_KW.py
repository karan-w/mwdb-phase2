import sys

sys.path.append(".")

import os
import logging
import numpy as np
from skimage import feature, exposure, color
import matplotlib.pyplot as plt
from matplotlib import image
import sklearn
from sklearn.datasets import fetch_olivetti_faces
from scipy.linalg import svd
from numpy import dot
from sklearn.decomposition import LatentDirichletAllocation
import json
import argparse
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from utils.image_reader import ImageReader
from utils.feature_models.cm import ColorMoments
from utils.feature_models.elbp import ExtendedLocalBinaryPattern
from utils.feature_models.hog import HistogramOfGradients
from utils.dimensionality_reduction.pca import PrincipalComponentAnalysis
from utils.dimensionality_reduction.svd import SingularValueDecomposition
from utils.dimensionality_reduction.lda import LatentDirichletAllocation
from utils.dimensionality_reduction.kmeans import KMeans
from utils.subject import Subject

COLOR_MOMENTS = 'CM'
EXTENDED_LBP = 'ELBP'
HISTOGRAM_OF_GRADIENTS = 'HOG'

PRINCIPAL_COMPONENT_ANALYSIS = 'PCA'
SINGULAR_VALUE_DECOMPOSITION = 'SVD'
LATENT_DIRICHLET_ALLOCATION = 'LDA'
KMEANS = 'kmeans'


logger = logging.getLogger(__name__)
logging.basicConfig(filename="logs/logs.log", filemode="w", level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S')

class Task1:
    def __init__(self):
        pass

    def setup_args_parser(self):
        parser = argparse.ArgumentParser()

        parser.add_argument('--model', type=str, choices=[COLOR_MOMENTS, EXTENDED_LBP, HISTOGRAM_OF_GRADIENTS], required=True)
        parser.add_argument('--x', type=str, required=True)
        parser.add_argument('--k', type=int, required=True)
        parser.add_argument('--dimensionality_reduction_technique', type=str, choices=[PRINCIPAL_COMPONENT_ANALYSIS, SINGULAR_VALUE_DECOMPOSITION, LATENT_DIRICHLET_ALLOCATION, KMEANS], required=True)
        parser.add_argument('--images_folder_path', type=str, required=True)
        parser.add_argument('--output_folder_path', type=str, required=True)
        
        return parser

    def log_args(self, args):
        logger.debug("Received the following arguments.")
        logger.debug(f'model - {args.model}')
        logger.debug(f'x - {args.x}')
        logger.debug(f'k - {args.k}')
        logger.debug(f'dimensionality_reduction_technique - {args.dimensionality_reduction_technique}')
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

    def reduce_dimensions(self, dimensionality_reduction_technique, images, k):
        if dimensionality_reduction_technique == PRINCIPAL_COMPONENT_ANALYSIS:
            return PrincipalComponentAnalysis().compute(images, k)
        elif dimensionality_reduction_technique == SINGULAR_VALUE_DECOMPOSITION:
            return SingularValueDecomposition().compute(images, k)
        elif dimensionality_reduction_technique == LATENT_DIRICHLET_ALLOCATION:
            return LatentDirichletAllocation().compute(images, k)
        elif dimensionality_reduction_technique == KMEANS:
            return KMeans().compute(images, k)
        else:
            raise Exception(f"Unknown dimensionality reduction technique - {dimensionality_reduction_technique}")

    def assign_images_to_subjects(self, images):
        subjects = []

        for index in range(0, 400, 10):
            subject = Subject(images[index:index+10])
            subjects.append(subject)

        return subjects

    def compute_subject_weight_matrix(self, subjects):
        return

    # I/O save
    def save_subject_weight_matrix(self, subject_weight_matrix):
        return

if __name__ == "__main__":
    task = Task1()
    parser = task.setup_args_parser()

    # model, x, k, dimensionality_reduction_technique, images_folder_path, output_folder_path
    args = parser.parse_args()
    task.log_args(args)

    image_reader = ImageReader()
    images = image_reader.get_images(args.images_folder_path, args.x)

    images = task.compute_feature_vectors(args.model, images)
    
    images = task.reduce_dimensions(args.dimensionality_reduction_technique, images, args.k)

    subjects = task.assign_images_to_subjects(images)

    # subject_weight_matrix = task.compute_subject_weight_matrix(subjects)

    # for image in images:
    #     print(image.reduced_feature_vector)

    # subject_weight_matrix = []
    # y = 0
    # while(y<40):
        
    #     image_data = []
    #     y+=1
    #     for i in range(1, 11):
    #         image_label = 'image-' + image_type + '-' + str(y) + '-' + str(i) + '.png' 
    #         image_data.append(image.imread(os.path.join(images_folder, image_label)))
    #     fv = Task1().features(feature_model, image_data)
    #     print(fv.shape)
    #     ls = Task1().dimension_red(reduction_method, fv, k_value)

    #     output_dict = dict.fromkeys(['Subject', 'Weight'])
    #     output_dict['Subject'] = y
    #     output_dict['Weight'] = ls.tolist()
    #     subject_weight_matrix.append(output_dict)

    # output_json = os.path.join(output_folder, 'data.json') # Create folder with timestamp and store in that

    # with open(output_json, 'w') as fp:
    #     for dictionary in subject_weight_matrix:
    #         json.dump(dictionary, fp, indent=4)

# TODO: Check all dimensionality reduction techniques
# TODO: Generalize code 
# TODO: Create output folder with timestamp and store in that
