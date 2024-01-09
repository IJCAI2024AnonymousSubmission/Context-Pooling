import os
import torch
from scipy.sparse import csr_matrix
import numpy as np
from collections import defaultdict
import pickle

from collections import defaultdict
import os
import networkx as nx
import pickle
from itertools import chain, combinations


class DataLoader:
    def __init__(self, task_dir):
        self.task_dir = task_dir

        with open(os.path.join(task_dir, 'entities.txt')) as f:
            self.entity2id = dict()
            n_ent = 0
            for line in f:
                entity = line.strip()
                self.entity2id[entity] = n_ent
                n_ent += 1

        with open(os.path.join(task_dir, 'relations.txt')) as f:
            self.relation2id = dict()
            n_rel = 0
            for line in f:
                relation = line.strip()
                self.relation2id[relation] = n_rel
                n_rel += 1

        self.n_ent = n_ent
        self.n_rel = n_rel

        self.filters = defaultdict(lambda:set())

        self.fact_triple  = self.read_triples('facts.txt')
        self.train_triple = self.read_triples('train.txt')
        self.valid_triple = self.read_triples('valid.txt')
        self.test_triple  = self.read_triples('test.txt')
    
        # add inverse
        self.fact_data  = self.double_triple(self.fact_triple)
        self.train_data = np.array(self.double_triple(self.train_triple))
        self.valid_data = self.double_triple(self.valid_triple)
        self.test_data  = self.double_triple(self.test_triple)

        self.load_graph(self.fact_data)
        self.load_test_graph(self.double_triple(self.fact_triple)+self.double_triple(self.train_triple))

        self.valid_q, self.valid_a = self.load_query(self.valid_data)
        self.test_q,  self.test_a  = self.load_query(self.test_data)

        self.n_train = len(self.train_data)
        self.n_valid = len(self.valid_q)
        self.n_test  = len(self.test_q)

        for filt in self.filters:
            self.filters[filt] = list(self.filters[filt])

        print('n_train:', self.n_train, 'n_valid:', self.n_valid, 'n_test:', self.n_test)

    def read_triples(self, filename):
        triples = []
        with open(os.path.join(self.task_dir, filename)) as f:
            for line in f:
                h, r, t = line.strip().split()
                h, r, t = self.entity2id[h], self.relation2id[r], self.entity2id[t]
                triples.append([h,r,t])
                self.filters[(h,r)].add(t)
                self.filters[(t,r+self.n_rel)].add(h)
        return triples

    def double_triple(self, triples):
        new_triples = []
        for triple in triples:
            h, r, t = triple
            new_triples.append([t, r+self.n_rel, h]) 
        return triples + new_triples

    def load_graph(self, triples):
        idd = np.concatenate([np.expand_dims(np.arange(self.n_ent),1), 2*self.n_rel*np.ones((self.n_ent, 1)), np.expand_dims(np.arange(self.n_ent),1)], 1)

        self.KG = np.concatenate([np.array(triples), idd], 0)
        self.n_fact = len(self.KG)
        self.M_sub = csr_matrix((np.ones((self.n_fact,)), (np.arange(self.n_fact), self.KG[:,0])), shape=(self.n_fact, self.n_ent))


    def load_test_graph(self, triples):
        idd = np.concatenate([np.expand_dims(np.arange(self.n_ent),1), 2*self.n_rel*np.ones((self.n_ent, 1)), np.expand_dims(np.arange(self.n_ent),1)], 1)

        self.tKG = np.concatenate([np.array(triples), idd], 0)
        self.tn_fact = len(self.tKG)
        self.tM_sub = csr_matrix((np.ones((self.tn_fact,)), (np.arange(self.tn_fact), self.tKG[:,0])), shape=(self.tn_fact, self.n_ent))

    def load_query(self, triples):
        triples.sort(key=lambda x:(x[0], x[1]))
        trip_hr = defaultdict(lambda:list())

        for trip in triples:
            h, r, t = trip
            trip_hr[(h,r)].append(t)
        
        queries = []
        answers = []
        for key in trip_hr:
            queries.append(key)
            answers.append(np.array(trip_hr[key]))
        return queries, answers

    def get_neighbors(self, nodes, mode='train'):
        if mode=='train':
            KG = self.KG
            M_sub = self.M_sub
        else:
            KG = self.tKG
            M_sub = self.tM_sub

        # nodes: n_node x 2 with (batch_idx, node_idx)
        node_1hot = csr_matrix((np.ones(len(nodes)), (nodes[:,1], nodes[:,0])), shape=(self.n_ent, nodes.shape[0]))
        edge_1hot = M_sub.dot(node_1hot)
        edges = np.nonzero(edge_1hot)
        sampled_edges = np.concatenate([np.expand_dims(edges[1],1), KG[edges[0]]], axis=1)     # (batch_idx, head, rela, tail)
        sampled_edges = torch.LongTensor(sampled_edges).cuda()

        # index to nodes
        head_nodes, head_index = torch.unique(sampled_edges[:,[0,1]], dim=0, sorted=True, return_inverse=True)
        tail_nodes, tail_index = torch.unique(sampled_edges[:,[0,3]], dim=0, sorted=True, return_inverse=True)

        sampled_edges = torch.cat([sampled_edges, head_index.unsqueeze(1), tail_index.unsqueeze(1)], 1)
       
        mask = sampled_edges[:,2] == (self.n_rel*2)
        _, old_idx = head_index[mask].sort()
        old_nodes_new_idx = tail_index[mask][old_idx]

        return tail_nodes, sampled_edges, old_nodes_new_idx

    def get_batch(self, batch_idx, steps=2, data='train'):
        if data=='train':
            return np.array(self.train_data)[batch_idx]
        if data=='valid':
            query, answer = np.array(self.valid_q), np.array(self.valid_a)
        if data=='test':
            query, answer = np.array(self.test_q), np.array(self.test_a)

        subs = []
        rels = []
        objs = []
        
        subs = query[batch_idx, 0]
        rels = query[batch_idx, 1]
        objs = np.zeros((len(batch_idx), self.n_ent))
        for i in range(len(batch_idx)):
            objs[i][answer[batch_idx[i]]] = 1
        return subs, rels, objs

    def shuffle_train(self,):
        fact_triple = np.array(self.fact_triple)
        train_triple = np.array(self.train_triple)
        all_triple = np.concatenate([fact_triple, train_triple], axis=0)
        n_all = len(all_triple)
        rand_idx = np.random.permutation(n_all)
        all_triple = all_triple[rand_idx]

        # increase the ratio of fact_data, e.g., 3/4->4/5, can increase the performance
        self.fact_data = self.double_triple(all_triple[:n_all*3//4].tolist())
        self.train_data = np.array(self.double_triple(all_triple[n_all*3//4:].tolist()))
        self.n_train = len(self.train_data)
        self.load_graph(self.fact_data)

class DistinctiveDataLoader(DataLoader):
    def __init__(self, task_dir, results_dir, do_generate, accuracy_threshold, recall_threshold):
        super(DistinctiveDataLoader, self).__init__(task_dir)
        relation2id = self.relation2id.copy()
        for r in relation2id:
            self.relation2id[r + '_inv'] = self.relation2id[r] + self.n_rel
        self.relation2id['idd'] = self.n_rel * 2
        self.id2entity = {}
        self.id2entity_ind = {}
        for e in self.entity2id:
            self.id2entity[self.entity2id[e]] = e

        self.subgraph_set = defaultdict(lambda: defaultdict(None))

        if do_generate:
            self.accuracy_tensor, \
            self.recall_tensor = self.generate_distinctive_neighbor(task_dir, results_dir, accuracy_threshold,
                                                                    recall_threshold)
        else:
            with open(os.path.join(results_dir, "distinctive_neighbors.pkl"), 'rb') as f:
                self.accuracy_tensor, \
                self.recall_tensor = torch.load(f)
    def inverse_relations(self,relations):
        idd= relations == self.n_rel * 2
        i_relation=(relations+(self.n_rel)) % (self.n_rel*2)
        i_relation[idd]=self.n_rel * 2
        return i_relation
    def generate_distinctive_neighbor(self, dir, save_dir, accuracy_threshold, recall_threshold):
        ind_dir = dir + '_ind'

        head2relation = defaultdict(lambda: defaultdict(int))
        tail2relation = defaultdict(lambda: defaultdict(int))
        relation2head = defaultdict(lambda: defaultdict(int))
        relation2tail = defaultdict(lambda: defaultdict(int))

        triplets = []
        relations = set()
        G_train = nx.DiGraph()
        with open(os.path.join(dir, 'train.txt')) as f:
            for line in f:
                h, r, t = line.strip().split()
                triplets.append([h, r, t])
                triplets.append([t, r + '_inv', h])
                G_train.add_edge(h, t, relation=r)
                G_train.add_edge(t, h, relation=str(r + '_inv'))
                relations.add(r)
                relations.add(r + '_inv')

                head2relation[h][r] += 1
                tail2relation[t][r] += 1
                relation2head[r][h] += 1
                relation2tail[r][t] += 1

                head2relation[t][r + '_inv'] += 1
                tail2relation[h][r + '_inv'] += 1
                relation2head[r + '_inv'][t] += 1
                relation2tail[r + '_inv'][h] += 1

        relation2neighbors = defaultdict(lambda: defaultdict(int))
        all_neighbors = defaultdict(int)
        for u in G_train.nodes():
            neighbors = frozenset(G_train[u][v]['relation'] for v in G_train[u])
            all_neighbors[neighbors] += 1
            for r in sorted(neighbors):
                relation2neighbors[r][neighbors] += 1

        neighbor_num = defaultdict(int)
        for r in sorted(relations):
            neighbor_num[r] = sum([relation2neighbors[r][n] for n in relation2neighbors[r]])

        accuracy_neighbors = defaultdict(set)
        recall_neighbors = defaultdict(set)
        for r in sorted(relations):
            for r2 in sorted(relations):
                cooccurrence = 0
                for n1 in relation2neighbors[r]:
                    if r2 in n1:
                        cooccurrence += relation2neighbors[r][n1]
                # cooccurrence2 = 0
                # for n2 in relation2neighbors[r2]:
                #     if r in n2:
                #         cooccurrence2+=relation2neighbors[r][n2]
                # assert cooccurrence2==cooccurrence
                if neighbor_num[r] > 0 and cooccurrence / neighbor_num[r] > accuracy_threshold:
                    accuracy_neighbors[r].add(r2)
                if neighbor_num[r2] > 0 and cooccurrence / neighbor_num[r2] > recall_threshold:
                    recall_neighbors[r].add(r2)

        accuracy_neighbors = {r: sorted(accuracy_neighbors[r]) for r in
                              sorted(relations)}
        recall_neighbors = {r: sorted(recall_neighbors[r]) for r in
                            sorted(relations)}

        # accuracy_neighbors = {self.relation2id[r]: torch.tensor(
        #     [self.relation2id[accuracy_neighbors[r][i]] for i in range(len(accuracy_neighbors[r]))]) for r in
        #     accuracy_neighbors}
        # recall_neighbors = {self.relation2id[r]: torch.tensor(
        #     [self.relation2id[recall_neighbors[r][i]] for i in range(len(recall_neighbors[r]))]) for
        #     r in recall_neighbors}

        accuracy_tensor_indices = torch.tensor([[self.relation2id[r2], self.relation2id[r1]] for r2 in
                                                sorted(relations) for r1 in accuracy_neighbors[r2]]).cuda()
        recall_tensor_indices = torch.tensor([[self.relation2id[r2], self.relation2id[r1]] for r2 in
                                              sorted(relations) for r1 in recall_neighbors[r2]]).cuda()
        accuracy_tensor = torch.zeros([len(self.relation2id) + 1, len(self.relation2id) + 1], dtype=torch.bool).cuda()
        recall_tensor = torch.zeros([len(self.relation2id) + 1, len(self.relation2id) + 1], dtype=torch.bool).cuda()
        if accuracy_tensor_indices.shape[0] > 0:
            accuracy_tensor[accuracy_tensor_indices[:, 0], accuracy_tensor_indices[:, 1]] = True
        if recall_tensor_indices.shape[0] > 0:
            recall_tensor[recall_tensor_indices[:, 0], recall_tensor_indices[:, 1]] = True

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(os.path.join(save_dir, "distinctive_neighbors.pkl"), "wb") as f:
            torch.save([accuracy_tensor, recall_tensor], f)
        return accuracy_tensor, recall_tensor

    def get_distinctive_neighbors(self, nodes, query_relations, distinctive_neighbors, head_index_full,
                                  tail_index_full,
                                  mode='train', complement=False):
        if mode == 'train':
            KG = self.KG
            M_sub = self.M_sub
        else:
            KG = self.tKG
            M_sub = self.tM_sub
        node_1hot = csr_matrix((np.ones(len(nodes)), (nodes[:, 1], nodes[:, 0])), shape=(self.n_ent, nodes.shape[0]))
        edge_1hot = M_sub.dot(node_1hot)
        edges = np.nonzero(edge_1hot)
        relations_next = KG[edges[0]][:, 1]

        relations_next = self.inverse_relations(relations_next)
        query_relations_next = query_relations[edges[1]]

        unique_relations, inverse_indices = np.unique(np.stack([relations_next, query_relations_next], axis=1),
                                                      axis=0,
                                                      return_inverse=True)

        idd_mask = torch.tensor(relations_next == self.n_rel * 2).cuda()
        distinctive_mask = distinctive_neighbors[unique_relations[:, 1], unique_relations[:, 0]]
        distinctive_mask = distinctive_mask[inverse_indices]
        distinctive_mask = distinctive_mask if complement == False else torch.logical_not(distinctive_mask)
        distinctive_mask = torch.logical_or(distinctive_mask, idd_mask)
        filtered_indexes = torch.where(distinctive_mask == True)[0].cpu().numpy()
        filtered_edges = (edges[0][filtered_indexes], edges[1][filtered_indexes])

        sampled_edges = np.concatenate([np.expand_dims(filtered_edges[1], 1), KG[filtered_edges[0]]],
                                       axis=1)  # (batch_idx, head, rela, tail)
        sampled_edges = torch.LongTensor(sampled_edges).cuda()
        # index to nodes
        head_nodes, head_index = torch.unique(sampled_edges[:, [0, 1]], dim=0, sorted=True, return_inverse=True)
        tail_nodes, tail_index = torch.unique(sampled_edges[:, [0, 3]], dim=0, sorted=True, return_inverse=True)

        head_index = head_index_full[sampled_edges[:, 0], sampled_edges[:, 1]]
        tail_index = tail_index_full[sampled_edges[:, 0], sampled_edges[:, 3]]


        # head_index=torch.tensor([head_indexes[tuple(sampled_edges[i][[0, 1]].cpu().tolist())] for i in range(len(sampled_edges))]).cuda()
        # tail_index = torch.tensor(
        #     [tail_indexes[tuple(sampled_edges[i][[0, 3]].cpu().tolist())] for i in range(len(sampled_edges))]).cuda()

        mask = sampled_edges[:, 2] == (self.n_rel * 2)
        _, old_idx = head_index[mask].sort()
        old_nodes_new_idx = tail_index[mask][old_idx]

        sampled_edges = torch.cat([sampled_edges, head_index.unsqueeze(1), tail_index.unsqueeze(1)], 1)
        return tail_nodes, sampled_edges, old_nodes_new_idx, relations_next