import os
import networkx as nx
from collections import defaultdict
from itertools import chain, combinations
import numpy as np
import pickle


def powerset(iterable,min_depth,max_depth):
    "powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3) without ()"
    s = sorted(list(iterable))
    return chain.from_iterable(combinations(s, r) for r in range(min_depth,max_depth))

def get_subgraph(G, neighbor_filter, center_node, query_relation, depth):
    G_sub=nx.DiGraph()
    neighbors=set(G[center_node][v]['relation'] for v in G[center_node])
    if query_relation not in neighbor_filter:
        return G_sub
    neighbor_similarity=[[f, len(neighbors.intersection(f))/len(neighbors.union(f))]for f in neighbor_filter[query_relation]]
    if len(neighbor_similarity)==0:
        return G_sub
    neighbor_rank=np.argsort([-ns[1] for ns in neighbor_similarity])
    filtered_relations=neighbor_similarity[neighbor_rank[0]][0]
    for v in G[center_node]:
        if G[center_node][v]['relation'] in filtered_relations:
            if depth >= 0:
                G_get=get_subgraph(G,neighbor_filter,v,G[center_node][v]['relation'],depth-1)
                G_sub=nx.compose(G_sub,G_get)
            G_sub.add_edge(center_node, v, relation=G[center_node][v]['relation'])
            G_sub.add_edge(v, center_node, relation=G[v][center_node]['relation'])
    return G_sub

if __name__ == "__main__":

    dir='inductive/data/fb237_v1'
    ind_dir=dir+'_ind'
    save_dir='inductive/results/fb237_v1'
    accuracy_threshold=0.5
    recall_threshold=0.1
    min_depth=1
    max_depth=5


    head2relation=defaultdict(lambda :defaultdict(int))
    tail2relation=defaultdict(lambda :defaultdict(int))
    relation2head=defaultdict(lambda :defaultdict(int))
    relation2tail=defaultdict(lambda :defaultdict(int))

    triplets = []
    relations=set()
    G_train = nx.DiGraph()
    with open(os.path.join(dir, 'train.txt')) as f:
        for line in f:
            h, r, t = line.strip().split()
            triplets.append([h, r, t])
            triplets.append([t,r+'_inv',h])
            G_train.add_edge(h, t, relation=r)
            G_train.add_edge(t, h, relation=str(r + '_inv'))
            relations.add(r)
            relations.add(r + '_inv')

            head2relation[h][r]+=1
            tail2relation[t][r]+=1
            relation2head[r][h]+=1
            relation2tail[r][t]+=1

            head2relation[t][r + '_inv']+=1
            tail2relation[h][r + '_inv']+=1
            relation2head[r + '_inv'][t]+=1
            relation2tail[r + '_inv'][h]+=1
    G_test = nx.DiGraph()
    with open(os.path.join(ind_dir, 'train.txt')) as f:
        for line in f:
            h, r, t = line.strip().split()
            triplets.append([h, r, t])
            triplets.append([t,r+'_inv',h])
            G_test.add_edge(h, t, relation=r)
            G_test.add_edge(t, h, relation=str(r + '_inv'))
            relations.add(r)
            relations.add(r + '_inv')

    relation2neighbors=defaultdict(lambda : defaultdict(int))
    all_neighbors=defaultdict(int)
    for u in G_train.nodes():
        neighbors=frozenset(G_train[u][v]['relation'] for v in G_train[u])
        all_neighbors[neighbors]+=1
        for r in neighbors:
            relation2neighbors[r][neighbors]+=1

    accuracy_neighbors=defaultdict(set)
    recall_neighbors=defaultdict(set)
    for r in sorted(relations):
        recall_neighbor_num = sum([relation2neighbors[r][n] for n in relation2neighbors[r]])
        for n1 in relation2neighbors[r]:
            other_neighbors=set(n1)-{r}
            if len(other_neighbors)!=0:
                potential_set=powerset(other_neighbors,min_depth,max_depth)
                for n2 in potential_set:
                    positive_neighbor_num=0
                    for n3 in relation2neighbors[r]:
                        if set(n2).issubset(set(n3)):
                            positive_neighbor_num+=relation2neighbors[r][n3]
                    accuracy_neighbor_num=0
                    for n4 in all_neighbors:
                        if set(n2).issubset(set(n4)):
                            accuracy_neighbor_num+=all_neighbors[n4]



                    if positive_neighbor_num/accuracy_neighbor_num>accuracy_threshold:
                        accuracy_neighbors[r].add(n2)
                    if positive_neighbor_num/recall_neighbor_num>recall_threshold:
                        recall_neighbors[r].add(n2)

    accuracy_recall_neighbors={r:accuracy_neighbors[r].intersection(recall_neighbors[r]) for r in accuracy_neighbors}


    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir,"distinctive_neighbors.pkl"),"wb") as f:
        pickle.dump([G_train,G_test,accuracy_neighbors,recall_neighbors,accuracy_recall_neighbors],f)



