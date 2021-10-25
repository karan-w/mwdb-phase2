from ctypes import Array
import json
import sys
sys.path.append(".")
from networkx.algorithms import similarity
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from numpy.random import randint
import argparse
import pandas as pd
from utils.output import Output
import os

class Task9:
    def __init__(self):
        pass

    def setup_args_parser(self):
        parser = argparse.ArgumentParser()

        parser.add_argument('--n', type=int, required=True)
        parser.add_argument('--m', type=int, required=True)
        parser.add_argument('--input_subjects', type=str, required=True)
        parser.add_argument('--sub_sub_matrix_path', type=str, required=True)
        parser.add_argument('--output_folder_path', type=str, required=True)
        
        return parser
    def Create_Similarity_Graph(self,Subjects,val,df,n=None):
        Created_Graph = nx.DiGraph()
        for x in Subjects:
            Created_Graph.add_node(x)
        for x in range(1,len(Subjects)+1,1):
            v = (df.loc[x])[1:] if n==None else (df.loc[x])[1:n]
            for node in v:
                Created_Graph.add_edge(x,node,weight = val[x][node])
        return Created_Graph

    def Draw_Graph(self,Similarity_Graph,Folder_Path):
        pos = nx.spring_layout(Similarity_Graph,scale=2)
        nx.draw_networkx_nodes(Similarity_Graph, pos)
        nx.draw_networkx_labels(Similarity_Graph, pos)
        nx.draw_networkx_edges(Similarity_Graph, pos, edge_color='r')
        labels = nx.get_edge_attributes(Similarity_Graph,'weight')
        nx.draw_networkx_edge_labels(Similarity_Graph,pos,edge_labels=labels)
        figure = plt.gcf()  # get current figure
        figure.set_size_inches(32, 16) # set figure's size manually to your full screen (32x18)
        plt.savefig(os.path.join(Folder_Path,'graph.png'), bbox_inches='tight',dpi = 200)

    def read_latent_semantics(self, latent_semantics_file):
        with open(latent_semantics_file, 'r') as f:
            latent_semantics = json.load(f)

        return latent_semantics['subject-subject-similarity-matrix']

    def Compute_Personalized_PageRank(self,Subjects,TransitionMatrix,SeedNodeSet):
        Transportation_Probability = 0.15
        Identity_Matrix = np.identity(len(Subjects), dtype=float)
        Coefficient_of_PI = Identity_Matrix - ((Transportation_Probability) * TransitionMatrix)
        ReSeeding_Vector = np.zeros(len(Subjects))

        P1_Teleportation_Discounting = np.zeros(len(Subjects))
        ReSeeding_Value = 1.0 / len(SeedNodeSet)
        for x in SeedNodeSet:
            ReSeeding_Vector[x - 1] = ReSeeding_Value
        PI_Value = np.dot(np.linalg.inv(Coefficient_of_PI), (1-Transportation_Probability) * ReSeeding_Vector)
        PI_Value = (PI_Value - min(PI_Value)) / (max(PI_Value) - min(PI_Value))

        P1_ReSeeding_Value = (1-Transportation_Probability) / len(SeedNodeSet)
        for x in Subjects:
            if x not in SeedNodeSet:
                P1_Teleportation_Discounting[x - 1] = PI_Value[x - 1] / (Transportation_Probability)
            else:
                P1_Teleportation_Discounting[x - 1] = (PI_Value[x - 1] - P1_ReSeeding_Value) / (Transportation_Probability)

        P2_Value = P1_Teleportation_Discounting / sum(P1_Teleportation_Discounting)
        Seed_Set_Significance = 0
        for x in SeedNodeSet:
            Seed_Set_Significance += P2_Value[x - 1]
        P3_Value=P2_Value
        for x in SeedNodeSet:
            P3_Value[x-1] = P1_Teleportation_Discounting[x-1]
        x = {}
        for i, d in enumerate(P3_Value):
            x[i+1] = d
        x = {k: v for k, v in sorted(x.items(), key=lambda item: item[1], reverse=True)}
        return list(x.keys())

    def Generate_Output(self,Computed_Ranked_Matrix,Subjects,n,Personalized_PageRank,m,Seed_Set):
        Top_Subjects = []
        Index_Subject = 'Top {0} Subjects'.format(n-1)
        for i in Subjects:
            Top_n_Subjects = dict.fromkeys(['Subject', Index_Subject])
            Top_n_Subjects['Subject'] = i
            Top_n_Subjects[Index_Subject] = ((np.array(Computed_Ranked_Matrix.loc[i]))[1:n]).tolist()
            Top_Subjects.append(Top_n_Subjects)
        # print(Top_Subjects)l3 = [x for x in l1 if x not in l2]
        output = {
            '{0} Most Similar Subjects Of Each Subject'.format(n-1) : Top_Subjects,
            '{0} Most Significant Subjects Including Seed Set - {1}'.format(m,Seed_Set) : Personalized_PageRank[:m],
            '{0} Most Significant Subjects Excluding Seed Set - {1}'.format(m,Seed_Set) : [x for x in Personalized_PageRank if x not in Seed_Set][:m]
        }
        return output

if __name__ == "__main__":
    task = Task9()
    parser = task.setup_args_parser()
    args = parser.parse_args()
    n = args.n + 1
    m = args.m
    input_subjects = [int(x) for x in args.input_subjects.split(',')]
    subjects = [x+1 for x in range(40)]
    input_file_path = args.sub_sub_matrix_path
    arr = np.round(task.read_latent_semantics(input_file_path),5)
    Subject_Subject_Similarity_Matrix = pd.DataFrame(arr, columns = subjects,index=subjects)
    Subject_Index_Descending_Matrix = pd.DataFrame(
            data=Subject_Subject_Similarity_Matrix.columns.values[np.argsort(-Subject_Subject_Similarity_Matrix.values, axis=1)],columns=subjects,index=subjects
        )
    n_Similarity_Graph = task.Create_Similarity_Graph(subjects,Subject_Subject_Similarity_Matrix,Subject_Index_Descending_Matrix,n)
    
    Complete_Similarity_Graph = task.Create_Similarity_Graph(subjects,Subject_Subject_Similarity_Matrix,Subject_Index_Descending_Matrix)

    Personalized_PageRank = task.Compute_Personalized_PageRank(subjects, Subject_Subject_Similarity_Matrix, input_subjects)
    output = task.Generate_Output(Subject_Index_Descending_Matrix,subjects,n,Personalized_PageRank,m,input_subjects)
    
    OUTPUT_FILE_NAME = 'output.json'
    timestamp_folder_path = Output().create_timestamp_folder(args.output_folder_path) # /Outputs/Task1 -> /Outputs/Task1/2021-10-21-23-25-23
    output_json_path = os.path.join(timestamp_folder_path, OUTPUT_FILE_NAME) # /Outputs/Task1/2021-10-21-23-25-23 -> /Outputs/Task1/2021-10-21-23-25-23/output.json
    Output().save_dict_as_json_file(output, output_json_path)
    task.Draw_Graph(n_Similarity_Graph,timestamp_folder_path)