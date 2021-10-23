import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from numpy.random import randint
from scipy.sparse.construct import random
import pandas as pd
if __name__ == "__main__":
    subjects = [x for x in range(40)]
    arr = [[round((x+1)/randint(1,10),2) for x in range(40)]for y in range(40)]
    # print(arr)
    G = nx.DiGraph()
    for x in subjects:
        G.add_node(x)
    val = pd.DataFrame(arr, columns = subjects)
    print(val)
    print('-----------')
    for x in range(len(subjects)):
        # val = (val.loc[x].sort_values(ascending=False))[:10]
        df = val
        val = pd.DataFrame(
        data=df.columns.values[np.argsort(-df.values, axis=1)]
    )
        # print(val)
        # print(len(val))
        # print(len(arr[0]))
        # break
        G.add_edge(x,x+1,weight = arr[x][x])

    print(G.nodes())
    print(G.edges())

    # pos=nx.get_node_attributes(G,'pos')
    pos = nx.spring_layout(G,scale=2)

    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, edge_color='r', arrows = True)
    # nx.draw(G,pos)
    labels = nx.get_edge_attributes(G,'weight')
    nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
    # plt.savefig("graph.png", dpi=1000)
    plt.show()