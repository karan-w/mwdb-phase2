import unittest
from unittest.case import expectedFailure   
import numpy as np

from utils.dimensionality_reduction.pca import PrincipalComponentAnalysis

# https://towardsdatascience.com/the-mathematics-behind-principal-component-analysis-fff2d7f4b643

class Test_TestPCA(unittest.TestCase):
    def test_pca(self):
        dataset_feature_vector = np.array(
            [
                [90, 60, 90], 
                [90, 90, 30],
                [60, 60, 60],
                [60, 60, 90],
                [30, 30, 30]
            ]
            , np.int32)
        reduced_dataset_feature_vector = PrincipalComponentAnalysis().compute_PCA(dataset_feature_vector, 2)
        print(reduced_dataset_feature_vector)
        

    def test_standardization_dataset_feature_vector(self):
        dataset_feature_vector = np.array(
            [
                [90, 60, 90], 
                [90, 90, 30],
                [60, 60, 60],
                [60, 60, 90],
                [30, 30, 30]
            ]
            , np.int32)
        
        expected_standardized_dataset_feature_vector = np.array(
            [
                [1.069, 0, 1.12],
                [1.069, 1.58, -1.12],
                [-0.26, 0, 0],
                [-0.26, 0, -1.12],
                [-1.604, -1.58, -1.12]

            ], np.float64
        )

        actual_standardized_dataset_feature_vector = PrincipalComponentAnalysis().standardize_dataset_feature_vector(dataset_feature_vector)
        print(actual_standardized_dataset_feature_vector)
        # np.testing.assert_array_almost_equal(expected_standardized_dataset_feature_vector, actual_standardized_dataset_feature_vector)

if __name__ == '__main__':
    unittest.main()
