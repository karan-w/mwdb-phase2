from re import sub
import sys

sys.path.append(".")

import os
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
import logging
import argparse
from scipy.spatial import distance
from sklearn import preprocessing
from sklearn.cluster import KMeans
from utils.image_reader import ImageReader
from utils.feature_vector import FeatureVector
from utils.feature_models.cm import ColorMoments
from utils.feature_models.elbp import ExtendedLocalBinaryPattern
from utils.feature_models.hog import HistogramOfGradients
from utils.dimensionality_reduction.pca import PrincipalComponentAnalysis
from utils.dimensionality_reduction.svd import SingularValueDecomposition
from utils.dimensionality_reduction.lda import LatentDirichletAllocation
from utils.dimensionality_reduction.kmeans import KMeans
from utils.subject import Subject
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

class Task4:
    def __init__(self):
        pass

    def setup_args_parser(self):
        parser = argparse.ArgumentParser()

        parser.add_argument('--model', type=str, choices=[COLOR_MOMENTS, EXTENDED_LBP, HISTOGRAM_OF_GRADIENTS], required=True)
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

    def get_distinct_subject_ids(self, images):
        distinct_subject_ids = set()
        for image in images:
            distinct_subject_ids.add(image.subject_id)
        return sorted(list(distinct_subject_ids))

    def assign_images_to_subjects(self, images):
        subject_ids = self.get_distinct_subject_ids(images)
        subjects = []

        # Group images for each distinct subject_id 
        for subject_id in subject_ids:
            subject_images = []
            for image in images:
                if(image.subject_id == subject_id):
                    subject_images.append(image)
            subject = Subject(subject_images)
            subject.create_subject_feature_vector(subject.images)
            subjects.append(subject)

        # Return all the distinct subjects as a list
        return subjects
    
    #computing subject-subject similarity matrix
    def compute_subject_similarity_matrix(self, subjects):
        subject_feature_vector = FeatureVector().create_subjects_feature_vector(subjects)
        subject_feature_vector_t = np.transpose(subject_feature_vector)
        subject_dist_matrix = [[0 for x in range(40)] for y in range(40)]

        #instead of multiplying D with D_t we use distance function to calculate distance between corresponsding values
        for i in range(len(subject_feature_vector)):
            for j in range(len(subject_feature_vector_t[0])):
                for k in range(len(subject_feature_vector_t)):
                    subject_dist_matrix[i][j] += distance.cityblock(subject_feature_vector[i][k], subject_feature_vector_t[k][j])
        
        subject_dist_matrix = np.array(subject_dist_matrix)
         
        #reshaping matrix to convert it to 1d array and then normalizing it
        subject_dist_matrix = subject_dist_matrix.reshape(1, len(subject_dist_matrix[0])*len(subject_dist_matrix[0]))

        # #we normazlize the distances
        subject_dist_matrix = preprocessing.normalize(subject_dist_matrix.reshape(1, -1), axis=1, norm='max')

        # #reshaping back to 40 x 40 matrix
        subject_dist_matrix = subject_dist_matrix.reshape(40, 40)

        #using 1-d_norm to calculate actual similairty
        subject_similarity_matrix = [[0 for x in range(40)] for y in range(40)]
        for i in range(len(subject_dist_matrix[0])):
            for j in range(len(subject_dist_matrix[1])):
                subject_similarity_matrix[i][j] = 1 - subject_dist_matrix[i][j]

        #convert to numpy array
        subject_similarity_matrix = np.array(subject_similarity_matrix)
    
        return subject_similarity_matrix

    def reduce_dimensions(self, dimensionality_reduction_technique, subject_similarity_matrix, k):
        if dimensionality_reduction_technique == PRINCIPAL_COMPONENT_ANALYSIS:
            return PrincipalComponentAnalysis().compute_subject_PCA(subject_similarity_matrix, k)
        elif dimensionality_reduction_technique == SINGULAR_VALUE_DECOMPOSITION:
            return SingularValueDecomposition().compute_subject_SVD(subject_similarity_matrix, k)
        elif dimensionality_reduction_technique == LATENT_DIRICHLET_ALLOCATION:
            return LatentDirichletAllocation().compute_subject_LDA(subject_similarity_matrix, k)
        elif dimensionality_reduction_technique == KMEANS:
            return KMeans().compute_subject_KMeans(subject_similarity_matrix, k)
        else:
            raise Exception(f"Unknown dimensionality reduction technique - {dimensionality_reduction_technique}")

    def build_output(self, args, images, drt_attributes, subjects, subject_weight_matrix, subject_similarity_matrix):
        # 1. Preprocess all variables/objects so they can be serialized
        # for image in images:
        #     image.matrix = None
        #     image.reduced_feature_vector = image.reduced_feature_vector.real.tolist()

        # for subject in subjects:
        #     subject.images = None
        #     subject.feature_vector = subject.feature_vector.real.tolist()
        #     subject.reduced_feature_vector = subject.reduced_feature_vector.real.tolist()

        # drt_attributes = self.preprocess_drt_attributes_for_output(args.dimensionality_reduction_technique, drt_attributes)

        #subject_weight_matrix = subject_weight_matrix.real.tolist()
        
        sorted_subject_weight_matrix = []
        for i in range(len(subject_weight_matrix[0])):
            subject_weight_pairs = dict.fromkeys(['Latent Semantic', 'Subjects', 'Weights'])
            subjects = []
            weights = []
            for j in range(len(subject_weight_matrix)):
                subjects.append(str(j))
                weights.append(subject_weight_matrix[j][i])
            subject_weight_pairs['Latent Semantic'] = i
            subject_weight_pairs['Weights'] = [x for x,_ in sorted(zip(weights,subjects), reverse=True)]
            subject_weight_pairs['Subjects'] = [x for _,x in sorted(zip(weights,subjects), reverse=True)]
            sorted_subject_weight_matrix.append(subject_weight_pairs)

        # 2. Prepare dictionary that should be JSONfied to store in JSON file
        output = {
            # args is not serializable
            # 'args': {
            #     'model': args.model,
            #     'x': args.x,
            #     'k': args.k,
            #     'dimensionality_reduction_technique': args.dimensionality_reduction_technique,
            #     'images_folder_path': args.images_folder_path,
            #     'output_folder_path': args.output_folder_path
            # },
            # 'images': images,
            # 'subjects': subjects,
            # 'drt_attributes': drt_attributes, 
            'subject_weight_matrix': sorted_subject_weight_matrix,
            'subject-subject-similarity-matrix': subject_similarity_matrix.real.tolist(),
        }
        return output

    def save_output(self, output, output_folder_path):
        OUTPUT_FILE_NAME = 'output.json'
        timestamp_folder_path = Output().create_timestamp_folder(output_folder_path) # /Outputs/Task1 -> /Outputs/Task1/2021-10-21-23-25-23
        output_json_path = os.path.join(timestamp_folder_path, OUTPUT_FILE_NAME) # /Outputs/Task1/2021-10-21-23-25-23 -> /Outputs/Task1/2021-10-21-23-25-23/output.json
        Output().save_dict_as_json_file(output, output_json_path)

def main():
    task = Task4()
    parser = task.setup_args_parser()
    
    # model, x, k, dimensionality_reduction_technique, images_folder_path, output_folder_path
    args = parser.parse_args()
    # task.log_args(args)

    image_reader = ImageReader()

    images = image_reader.get_all_images_in_folder(args.images_folder_path)

    images = task.compute_feature_vectors(args.model, images)

    subjects = task.assign_images_to_subjects(images)

    subject_similarity_matrix = task.compute_subject_similarity_matrix(subjects)

    subject_weight_matrix, drt_attributes = task.reduce_dimensions(args.dimensionality_reduction_technique, subject_similarity_matrix, args.k)
    
    output = task.build_output(args, images, drt_attributes, subjects, subject_weight_matrix, subject_similarity_matrix)

    task.save_output(output, args.output_folder_path)

if __name__ == "__main__":
    main()