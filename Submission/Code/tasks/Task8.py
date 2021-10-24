import sys
import os

sys.path.append(".")

import argparse 
import json

import numpy as np

from utils.graph.similarity_graph import SimilarityGraph
from utils.graph.ascos_pp import ASCOS_PP
from utils.output import Output
from networkx.linalg import graphmatrix

class Task8:
    def __init__(self):
        pass
        
    def setup_args_parser(self):
        parser = argparse.ArgumentParser()

        parser.add_argument('--subject_subject_similarity_matrix_filepath', type=str, required=True)
        parser.add_argument('--number_of_subjects', type=str, required=True)
        parser.add_argument('--n', type=int, required=True)
        parser.add_argument('--m', type=int, required=True)
        parser.add_argument('--output_folder_path', type=str, required=True)
        
        return parser

    def read_subject_subject_similarity_matrix(self, filepath):
        # subject_subject_similarity_matrix = np.array(
        #     [
        #         [1.00, 0.57, 0.51, 0.26, 0.31, 0.33], 
        #         [0.57, 1.00, 0.54, 0.25, 0.31, 0.43],
        #         [0.51, 0.54, 1.00, 0.19, 0.25, 0.36],
        #         [0.26, 0.25, 0.19, 1.00, 0.50, 0.38],
        #         [0.31, 0.31, 0.25, 0.50, 1.00, 0.56],
        #         [0.33, 0.43, 0.36, 0.38, 0.56, 1.00]
        #     ], np.float64)
        
        # number_of_subjects = 6

        subject_subject_similarity_matrix = None
        number_of_subjects = None

        # Open the JSON file from the filepath
        with open(filepath) as json_file:
            data = json.load(json_file) # Loads the JSON as a dictionary
            subject_subject_similarity_matrix = data['subject-subject-similarity-matrix']
            number_of_subjects = 40

        # # Convert list of list to np ndarray

        return subject_subject_similarity_matrix, number_of_subjects

    def find_subjects_significance(self, S):
        S = S[~np.eye(S.shape[0],dtype=bool)].reshape(S.shape[0],-1) # Remove the diagonals 6 * 5
        subjects_significance = np.sum(S, axis=1) # 6 * 1 - rows
        return subjects_significance

    # S matrix of ASCOS++ computation
    def find_m_significant_subjects(self, subjects_significance, m):
        subjects_significance_ascending_ranking = np.argsort(subjects_significance) # Ranking from lowest to highest
        subjects_significance_descending_ranking = np.flip(subjects_significance_ascending_ranking)
        return subjects_significance_descending_ranking[:m] # slice top m
    
    def save_output(self, args, subjects_significance, m_significant_subjects, similarity_graph):
        
        # will be sorted
        ranked_subjects_significance = []
        
        for subject_id, subject_significance in enumerate(subjects_significance):
            ranked_subjects_significance.append((subject_id, subject_significance))

        ranked_subjects_significance.sort(key=lambda x:x[1])
        ranked_subjects_significance.reverse()
        similarity_graph_adjacency_matrix = graphmatrix.adjacency_matrix(similarity_graph.graph).toarray().tolist()
        output = {
            'args': {
                'subject_subject_similarity_matrix_filepath': args.subject_subject_similarity_matrix_filepath,
                'number_of_subjects': args.number_of_subjects,
                'n': args.n,
                'm': args.m,
                'output_folder_path': args.output_folder_path
            },
            'm_significant_subjects': m_significant_subjects.tolist(),
            'ranked_subjects_significance': ranked_subjects_significance,
            'similarity_graph': {
                'adjacency_matrix': similarity_graph_adjacency_matrix
            }
        }

        OUTPUT_FILE_NAME = 'output.json'
        timestamp_folder_path = Output().create_timestamp_folder(args.output_folder_path) # /Outputs/Task8-> /Outputs/Task8/2021-10-21-23-25-23

        similarity_graph.save(timestamp_folder_path)

        output_json_path = os.path.join(timestamp_folder_path, OUTPUT_FILE_NAME) # /Outputs/Task8/2021-10-21-23-25-23 -> /Outputs/Task8/2021-10-21-23-25-23/output.json
        Output().save_dict_as_json_file(output, output_json_path)

def main():
    task = Task8()
    parser = task.setup_args_parser()

    # subject_subject_similarity_matrix_filepath, number_of_subjects, n, m, output_folder_path 
    args = parser.parse_args()

    subject_subject_similarity_matrix, number_of_subjects = task.read_subject_subject_similarity_matrix(args.subject_subject_similarity_matrix_filepath)

    similarity_graph = SimilarityGraph()
    similarity_graph.create(number_of_subjects, subject_subject_similarity_matrix, args.n)
    

    ascos_pp = ASCOS_PP(similarity_graph)
    ascos_pp.compute_P()
    ascos_pp.compute_Q()
    ascos_pp.compute_S(0.5)

    subjects_significance = task.find_subjects_significance(ascos_pp.S)

    m_significant_subjects = task.find_m_significant_subjects(subjects_significance, args.m)

    task.save_output(args, subjects_significance, m_significant_subjects, similarity_graph)

if __name__ == "__main__":
    main()