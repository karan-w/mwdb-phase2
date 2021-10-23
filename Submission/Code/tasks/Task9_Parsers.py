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
import random

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

        print(TransitionMatrix)
        Transportation_Probability = 0.15
        Identity_Matrix = np.identity(len(Subjects), dtype=float)
        # Coefficient_of_PI = Identity_Matrix - ((1 - Transportation_Probability) * TransitionMatrix)
        Coefficient_of_PI = Identity_Matrix - (Transportation_Probability * TransitionMatrix)

        ReSeeding_Vector = np.zeros(len(Subjects))
        P1_Teleportation_Discounting = np.zeros(len(Subjects))
        ReSeeding_Value = 1.0 / len(SeedNodeSet)
        for x in SeedNodeSet:
            ReSeeding_Vector[x - 1] = ReSeeding_Value
        ReSeeding_Vector = (1 - Transportation_Probability) * ReSeeding_Vector

        PI_Value = np.dot(np.linalg.inv(Coefficient_of_PI), ReSeeding_Vector)
        print(PI_Value)
        return PI_Value
        # P1_ReSeeding_Value = Transportation_Probability / len(SeedNodeSet)
        # for x in Subjects:
        #     if x not in SeedNodeSet:
        #         P1_Teleportation_Discounting[x - 1] = PI_Value[x - 1] / (1 - Transportation_Probability)
        #     else:
        #         P1_Teleportation_Discounting[x - 1] = (PI_Value[x - 1] - P1_ReSeeding_Value) / (
        #         1 - Transportation_Probability)
        # # print(P1_Teleportation_Discounting)
        # P1_Teleportation_Discounting = P1_Teleportation_Discounting / sum(P1_Teleportation_Discounting)
        # # print(P1_Teleportation_Discounting)
        # Seed_Set_Significance = 0
        # for x in SeedNodeSet:
        #     Seed_Set_Significance += P1_Teleportation_Discounting[x - 1]
        # print(P1_Teleportation_Discounting)
        # print(Seed_Set_Significance)
        # return


    def pagerank_test(self,Subjects,TransitionMatrix,SeedNodeSet):

        Transportation_Probability = 0.85
        Identity_Matrix = np.identity(len(Subjects), dtype=float)
        Coefficient_of_PI = Identity_Matrix - ((1 - Transportation_Probability) * TransitionMatrix)
        ReSeeding_Vector = np.zeros(len(Subjects))

        P1_Teleportation_Discounting = np.zeros(len(Subjects))
        ReSeeding_Value = 1.0 / len(SeedNodeSet)
        for x in SeedNodeSet:
            ReSeeding_Vector[x - 1] = ReSeeding_Value
        PI_Value = np.dot(np.linalg.inv(Coefficient_of_PI), Transportation_Probability * ReSeeding_Vector)
        PI_Value = (PI_Value - min(PI_Value)) / (max(PI_Value) - min(PI_Value))

        print("PI value \n")
        print(PI_Value)

        P1_ReSeeding_Value = Transportation_Probability / len(SeedNodeSet)
        for x in Subjects:
            if x not in SeedNodeSet:
                P1_Teleportation_Discounting[x - 1] = PI_Value[x - 1] / (1 - Transportation_Probability)
            else:
                P1_Teleportation_Discounting[x - 1] = (PI_Value[x - 1] - P1_ReSeeding_Value) / (
                1 - Transportation_Probability)

        print("P1 value \n")
        print(P1_Teleportation_Discounting)

        # P1_Teleportation_Discounting = (P1_Teleportation_Discounting - min(P1_Teleportation_Discounting))/(max(P1_Teleportation_Discounting)-min(P1_Teleportation_Discounting))
        P2_Value = P1_Teleportation_Discounting / sum(P1_Teleportation_Discounting)
        # print(P1_Teleportation_Discounting)
        print(P2_Value)
        Seed_Set_Significance = 0
        for x in SeedNodeSet:
            Seed_Set_Significance += P2_Value[x - 1]

        print("P2 value \n")
        print(sum(P2_Value))
        print(Seed_Set_Significance)

        # x={}
        # for i,d in enumerate(PI_Value):
        #     x[i]=d


        P3_Value=P2_Value
        # for i,p in enumerate(PI_Value):
        #     if i in SeedNodeSet:
        #         P3_Value[i-1] = P1_Teleportation_Discounting[i-1]

        for x in SeedNodeSet:
            P3_Value[x-1] = P1_Teleportation_Discounting[x-1]

        x = {}
        for i, d in enumerate(P3_Value):
            x[i+1] = d

        x = {k: v for k, v in sorted(x.items(), key=lambda item: item[1], reverse=True)}
        print(x.keys())
        return

    def Save_DataFrame_To_JSON(self,data):
        return data.to_json('./Task9.json', orient='index')


    def pppr(self,D):
        print("D\n",D)
        x = nx.pagerank(D,personalization={1:.33,2:.33,3:.33})
        x = {k: v for k, v in sorted(x.items(), key=lambda item: item[1],reverse=True)}
        print(x.keys())


if __name__ == "__main__":
    task = Task9()
    parser = task.setup_args_parser()
    args = parser.parse_args()
    n = args.n + 1
    m = args.m
    input_subjects = [int(x) for x in args.input_subjects.split(',')]
    subjects = [x+1 for x in range(40)]

    # arr = [[round((x+1)/randint(1,10),2) for x in range(40)]for y in range(40)]

    # arr = [[random.random() for x in range(40) if x>y]for y in range(40)]
    # print(arr)

    identity = np.identity(40,dtype=float)
    arr = np.random.rand(40, 40)
    arr = np.tril(arr,-1) + np.tril(arr, -1).T + identity


    Subject_Subject_Similarity_Matrix = pd.DataFrame(arr, columns = subjects,index=subjects)
    Subject_Index_Descending_Matrix = pd.DataFrame(
            data=Subject_Subject_Similarity_Matrix.columns.values[np.argsort(-Subject_Subject_Similarity_Matrix.values, axis=1)],columns=subjects,index=subjects
        )
    # n_Similarity_Graph = task.Create_Similarity_Graph(subjects,Subject_Subject_Similarity_Matrix,Subject_Index_Descending_Matrix,n)
    # task.Draw_Graph(n_Similarity_Graph)
    Complete_Similarity_Graph = task.Create_Similarity_Graph(subjects,Subject_Subject_Similarity_Matrix,Subject_Index_Descending_Matrix)
    # task.Compute_Personalized_PageRank(subjects,Subject_Subject_Similarity_Matrix,input_subjects)

    task.pagerank_test(subjects, Subject_Subject_Similarity_Matrix, input_subjects)
    task.pppr(Complete_Similarity_Graph)

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