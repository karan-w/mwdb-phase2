import sys
sys.path.append(".")
import os
from zipfile import ZipFile
from os import listdir
from os.path import isfile, join
from pathlib import Path
from PIL import Image

from utils.image import Image as custom_img

from scipy.spatial import distance
import matplotlib.image as img
import numpy as np
# from tasks.Task1_test import Task1
from timeit import default_timer as timer
from datetime import timedelta
from sklearn import preprocessing
import argparse

import logging

from utils.image_reader import ImageReader
from utils.feature_models.cm import ColorMoments
from utils.feature_models.elbp import ExtendedLocalBinaryPattern
from utils.feature_models.hog import HistogramOfGradients
from utils.dimensionality_reduction.pca import PrincipalComponentAnalysis
from utils.dimensionality_reduction.svd import SingularValueDecomposition
from utils.dimensionality_reduction.lda import LatentDirichletAllocation
from utils.dimensionality_reduction.kmeans import KMeans
# from utils.subject import Subject

from utils.feature_models.cm import ColorMoments
from utils.feature_models.elbp import ExtendedLocalBinaryPattern
from utils.feature_models.hog import HistogramOfGradients

COLOR_MOMENTS = 'CM'
EXTENDED_LBP = 'ELBP'
HISTOGRAM_OF_GRADIENTS = 'HOG'

PRINCIPAL_COMPONENT_ANALYSIS = 'PCA'
SINGULAR_VALUE_DECOMPOSITION = 'SVD'
LATENT_DIRICHLET_ALLOCATION = 'LDA'
KMEANS = 'kmeans'

logger = logging.getLogger(__name__)
logging.basicConfig(filename="logs/logs.log", filemode="w", level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S')

class Task3:
    def setup_args_parser(self):
        prsr = argparse.ArgumentParser()
        prsr.add_argument('--model', type=str, choices=[COLOR_MOMENTS, EXTENDED_LBP, HISTOGRAM_OF_GRADIENTS],
                            required=True)
        prsr.add_argument('--k', type=int, required=True)
        prsr.add_argument('--dimensionality_reduction_technique', type=str,
                            choices=[PRINCIPAL_COMPONENT_ANALYSIS, SINGULAR_VALUE_DECOMPOSITION,
                                     LATENT_DIRICHLET_ALLOCATION, KMEANS], required=True)
        prsr.add_argument('--images_folder_path', type=str, required=True)
        prsr.add_argument('--output_folder_path', type=str, required=True)

        return prsr

    # def log_args(self, args):
    #     logger.debug("Received the following arguments.")
    #     logger.debug(f'model - {args.model}')
    #     logger.debug(f'x - {args.x}')
    #     logger.debug(f'k - {args.k}')
    #     logger.debug(f'dimensionality_reduction_technique - {args.dimensionality_reduction_technique}')
    #     logger.debug(f'images_folder_path - {args.images_folder_path}')
    #     logger.debug(f'output_folder_path - {args.output_folder_path}')


    def compute_feature_vectors(self, feature_model, image):
        if feature_model == COLOR_MOMENTS:
            return ColorMoments().compute2(image)
        elif feature_model == EXTENDED_LBP:
            return ExtendedLocalBinaryPattern().compute2(image)
        elif feature_model == HISTOGRAM_OF_GRADIENTS:
            return HistogramOfGradients().compute2(image)
        else:
            raise Exception(f"Unknown feature model - {feature_model}")

    def reduce_dimensions(self, dimensionality_reduction_technique, images, k):
        if dimensionality_reduction_technique == PRINCIPAL_COMPONENT_ANALYSIS:
            return PrincipalComponentAnalysis().compute2(images, k)
        elif dimensionality_reduction_technique == SINGULAR_VALUE_DECOMPOSITION:
            return SingularValueDecomposition().compute2(images, k)
        elif dimensionality_reduction_technique == LATENT_DIRICHLET_ALLOCATION:
            return LatentDirichletAllocation().compute2(images, k)
        elif dimensionality_reduction_technique == KMEANS:
            return KMeans().compute2(images, k)
        else:
            raise Exception(f"Unknown dimensionality reduction technique - {dimensionality_reduction_technique}")

if __name__=="__main__":
    type_dict={"cc":0, "con":1, "emboss":2, "jitter":3, "neg":4, "noise01":5, "noise02":6, "original":7, "poster":8, "rot":9, "smooth":10, "stipple":11}

    task = Task3()
    parser = task.setup_args_parser()
    args = parser.parse_args()

    image_reader = ImageReader()
    type_matrix = []
    type_feature_mat = [[[] for i in range(400)] for j in range(12)]
    # type_feature_mat = [[] for i in range(12)]

    for i in range(13):
        type_matrix.append([])

    print(np.shape(type_feature_mat))
    start = timer()
    fv_size = 0

    files = [f for f in os.listdir(args.images_folder_path) if isfile(join(args.images_folder_path, f))]
    for file in files:
        image_details = file.replace('.png','').split('-')
        image_type = image_details[1]
        image_index = (int(image_details[2])-1)*10 + int(image_details[3])-1
        subject_id = image_details[2]
        image_id = image_details[3]

        imgg = image_reader.get_image(args.images_folder_path,image_type,subject_id,image_id)

        # image_data=[img.imread("all/"+file)]
        fv = task.compute_feature_vectors(args.model,imgg)

        # print("fv \n",fv)
        print("-----------------------------")
        print(np.shape(fv.feature_vector))
        # fv = Task1().features(feature_model, image_data)
        fv.feature_vector = np.array(fv.feature_vector)
        if fv_size==0:
            fv_size = np.shape(fv.feature_vector.flatten())
        type_feature_mat[type_dict[image_type]][image_index] = fv.feature_vector.flatten()
    print(np.shape(type_feature_mat[2]))

    print(fv_size)
    z = np.zeros(fv_size, dtype=object)


    for i in range(len(type_feature_mat)):
        for j in range(len(type_feature_mat[0])):
            if np.shape(type_feature_mat[i][j]) == (0,):
                type_feature_mat[i][j] = z

    #
    # =================================START=====================================

    print("type ",np.shape(type_feature_mat))
    type_type_mat = [[0 for x in range(len(type_dict))] for y in range(len(type_dict))]

    for i in range(len(type_feature_mat)):
        for j in range(len(type_feature_mat)):
            print(np.shape(type_feature_mat[i][j]))
            if np.shape(type_feature_mat[i][j][0])==1:
                print(type_feature_mat[i][j])
            print(np.shape(np.concatenate(type_feature_mat[i]).flat))
            print(np.shape(np.concatenate(type_feature_mat[j]).flat))
            type_type_mat[i][j] = distance.cityblock(np.concatenate(type_feature_mat[i]).flat,np.concatenate(type_feature_mat[j]).flat)

    print(type_type_mat)

    max_value = max(max(type_type_mat, key=max))

    for i in range(len(type_dict)):
        # type_type_mat[i] = np.array(type_type_mat[i],dtype=float)
        for j in range(len(type_dict)):
            type_type_mat[i][j] = 1 - (type_type_mat[i][j] / max_value)
    type_type_mat = np.array(type_type_mat,dtype=float)
    print(type_type_mat)

    dr = task.reduce_dimensions(args.dimensionality_reduction_technique, type_type_mat, args.k)

    # dr = Task1().dimension_red(reduction_method, type_type_mat, k_value)
    print("dr \n",dr)
        # =================================END=====================================
        #

    # type_feature_mat_t = np.transpose(type_feature_mat)
    #
    # type_type_mat = [ [0 for x in range(len(type_dict))]for y in range(len(type_dict))]
    # type_type_sim = type_type_mat
    # max_val = 0
    # for i in range(len(type_feature_mat)):
    #     for j in range(len(type_feature_mat_t[0])):
    #         for k in range(len(type_feature_mat_t)):
    #             type_type_mat[i][j] += distance.cityblock(type_feature_mat[i][k], type_feature_mat_t[k][j])

    # max_value = max(max(type_type_mat, key=max))
    #
    # for i in range(len(type_dict)):
    #     for j in range(len(type_dict)):
    #         type_type_sim[i][j] = 1 - (type_type_mat[i][j]/max_value)
    # # print(type_type_sim)
    #
    # dr = Task1().dimension_red(reduction_method,type_type_sim,k_value)
    # print(dr)
    end = timer()
    print(timedelta(seconds=end-start))