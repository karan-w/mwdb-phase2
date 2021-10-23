from ctypes import Array
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

    def Draw_Graph(self,Similarity_Graph):
        pos = nx.spring_layout(Similarity_Graph,scale=2)
        nx.draw_networkx_nodes(Similarity_Graph, pos)
        nx.draw_networkx_labels(Similarity_Graph, pos)
        nx.draw_networkx_edges(Similarity_Graph, pos, edge_color='r', arrows = True)
        labels = nx.get_edge_attributes(Similarity_Graph,'weight')
        nx.draw_networkx_edge_labels(Similarity_Graph,pos,edge_labels=labels)
        # plt.savefig("graph.png")
        plt.show()

    def Compute_Personalized_PageRank(self,Subjects,TransitionMatrix,SeedNodeSet):
        Transportation_Probability = 0.15
        Identity_Matrix = np.identity(len(Subjects), dtype = float)
        Coefficient_of_PI = Identity_Matrix - ((1-Transportation_Probability)*TransitionMatrix)
        ReSeeding_Vector = np.zeros(len(Subjects))
        ReSeeding_Value = 1.0/len(SeedNodeSet)
        for x in SeedNodeSet:
            ReSeeding_Vector[x] = ReSeeding_Value
        ReSeeding_Vector = Transportation_Probability * ReSeeding_Vector
        PI_Value = np.dot(np.linalg.inv(Coefficient_of_PI),ReSeeding_Vector)
        print(PI_Value)
        return

    def Save_DataFrame_To_JSON(self,data):
        return data.to_json('./Task9.json', orient='index')

if __name__ == "__main__":
    task = Task9()
    parser = task.setup_args_parser()
    args = parser.parse_args()
    n = args.n + 1
    m = args.m
    input_subjects = [int(x) for x in args.input_subjects.split(',')]
    subjects = [x+1 for x in range(40)]
    arr = [[round((x+1)/randint(1,10),2) for x in range(40)]for y in range(40)]
    Subject_Subject_Similarity_Matrix = pd.DataFrame(arr, columns = subjects,index=subjects)
    Subject_Index_Descending_Matrix = pd.DataFrame(
            data=Subject_Subject_Similarity_Matrix.columns.values[np.argsort(-Subject_Subject_Similarity_Matrix.values, axis=1)],columns=subjects,index=subjects
        )
    n_Similarity_Graph = task.Create_Similarity_Graph(subjects,Subject_Subject_Similarity_Matrix,Subject_Index_Descending_Matrix,n)
    task.Draw_Graph(n_Similarity_Graph)
    Complete_Similarity_Graph = task.Create_Similarity_Graph(subjects,Subject_Subject_Similarity_Matrix,Subject_Index_Descending_Matrix)
    task.Compute_Personalized_PageRank(subjects,Subject_Subject_Similarity_Matrix,input_subjects)
    # for u, v, weight in Complete_Similarity_Graph.edges(data="weight"):
    #     print(u,v,weight)
    # Subject_Subject_Similarity_Matrix.to_json(args.output_folder_path+'/Task9.json')
    # output = {
    #         'Subject_Subject_Similarity_Matrix': Subject_Subject_Similarity_Matrix.to_json(orient='split') ,
    #         'Most_Similar_Subjects': Subject_Index_Descending_Matrix.to_json(orient='index') 
    #     }
    # OUTPUT_FILE_NAME = 'Task9.json'
    # timestamp_folder_path = Output().create_timestamp_folder(args.output_folder_path) # /Outputs/Task1 -> /Outputs/Task1/2021-10-21-23-25-23
    # output_json_path = os.path.join(timestamp_folder_path, OUTPUT_FILE_NAME) # /Outputs/Task1/2021-10-21-23-25-23 -> /Outputs/Task1/2021-10-21-23-25-23/output.json
    # Output().save_dict_as_json_file(output, output_json_path)