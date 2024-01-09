import os
import torch
from scipy.sparse import csr_matrix
import numpy as np
from collections import defaultdict
import pickle

from collections import defaultdict
import os
import networkx as nx


class DataLoader:
    def __init__(self, task_dir):
        self.trans_dir = task_dir
        self.ind_dir = task_dir + '_ind'

        with open(os.path.join(task_dir, 'entities.txt')) as f:
            self.entity2id = dict()
            for line in f:
                entity, eid = line.strip().split()
                self.entity2id[entity] = int(eid)

        with open(os.path.join(task_dir, 'relations.txt')) as f:
            self.relation2id = dict()
            id2relation = []
            for line in f:
                relation, rid = line.strip().split()
                self.relation2id[relation] = int(rid)
                id2relation.append(relation)

        with open(os.path.join(self.ind_dir, 'entities.txt')) as f:
            self.entity2id_ind = dict()
            for line in f:
                entity, eid = line.strip().split()
                self.entity2id_ind[entity] = int(eid)

        for i in range(len(self.relation2id)):
            id2relation.append(id2relation[i] + '_inv')
        id2relation.append('idd')
        self.id2relation = id2relation

        self.n_ent = len(self.entity2id)
        self.n_rel = len(self.relation2id)
        self.n_ent_ind = len(self.entity2id_ind)

        self.tra_train = self.read_triples(self.trans_dir, 'train.txt')
        self.tra_valid = self.read_triples(self.trans_dir, 'valid.txt')
        self.tra_test = self.read_triples(self.trans_dir, 'test.txt')
        self.ind_train = self.read_triples(self.ind_dir, 'train.txt', 'inductive')
        self.ind_valid = self.read_triples(self.ind_dir, 'valid.txt', 'inductive')
        self.ind_test = self.read_triples(self.ind_dir, 'test.txt', 'inductive')

        self.val_filters = self.get_filter('valid')
        self.tst_filters = self.get_filter('test')

        for filt in self.val_filters:
            self.val_filters[filt] = list(self.val_filters[filt])
        for filt in self.tst_filters:
            self.tst_filters[filt] = list(self.tst_filters[filt])

        self.tra_KG, self.tra_sub = self.load_graph(self.tra_train)
        self.ind_KG, self.ind_sub = self.load_graph(self.ind_train, 'inductive')

        self.tra_train = np.array(self.tra_valid)
        self.tra_val_qry, self.tra_val_ans = self.load_query(self.tra_test)
        self.ind_val_qry, self.ind_val_ans = self.load_query(self.ind_valid)
        self.ind_tst_qry, self.ind_tst_ans = self.load_query(self.ind_test)
        self.valid_q, self.valid_a = self.tra_val_qry, self.tra_val_ans
        self.test_q, self.test_a = self.ind_val_qry + self.ind_tst_qry, self.ind_val_ans + self.ind_tst_ans

        self.n_train = len(self.tra_train)
        self.n_valid = len(self.valid_q)
        self.n_test = len(self.test_q)

        print('n_train:', self.n_train, 'n_valid:', self.n_valid, 'n_test:', self.n_test)

    def read_triples(self, directory, filename, mode='transductive'):
        triples = []
        with open(os.path.join(directory, filename)) as f:
            for line in f:
                h, r, t = line.strip().split()
                if mode == 'transductive':
                    h, r, t = self.entity2id[h], self.relation2id[r], self.entity2id[t]
                else:
                    h, r, t = self.entity2id_ind[h], self.relation2id[r], self.entity2id_ind[t]

                triples.append([h, r, t])
                triples.append([t, r + self.n_rel, h])
        return triples

    def load_graph(self, triples, mode='transductive'):
        n_ent = self.n_ent if mode == 'transductive' else self.n_ent_ind

        KG = np.array(triples)
        idd = np.concatenate([np.expand_dims(np.arange(n_ent), 1), 2 * self.n_rel * np.ones((n_ent, 1)),
                              np.expand_dims(np.arange(n_ent), 1)], 1)
        KG = np.concatenate([KG, idd], 0)

        n_fact = KG.shape[0]

        M_sub = csr_matrix((np.ones((n_fact,)), (np.arange(n_fact), KG[:, 0])), shape=(n_fact, n_ent))
        return KG, M_sub

    def load_query(self, triples):
        triples.sort(key=lambda x: (x[0], x[1]))
        trip_hr = defaultdict(lambda: list())

        for trip in triples:
            h, r, t = trip
            trip_hr[(h, r)].append(t)

        queries = []
        answers = []
        for key in trip_hr:
            queries.append(key)
            answers.append(np.array(trip_hr[key]))
        return queries, answers

    def get_neighbors(self, nodes, mode='transductive'):
        # nodes: n_node x 2 with (batch_idx, node_idx)

        if mode == 'transductive':
            KG = self.tra_KG
            M_sub = self.tra_sub
            n_ent = self.n_ent
        else:
            KG = self.ind_KG
            M_sub = self.ind_sub
            n_ent = self.n_ent_ind

        node_1hot = csr_matrix((np.ones(len(nodes)), (nodes[:, 1], nodes[:, 0])), shape=(n_ent, nodes.shape[0]))
        edge_1hot = M_sub.dot(node_1hot)
        edges = np.nonzero(edge_1hot)
        sampled_edges = np.concatenate([np.expand_dims(edges[1], 1), KG[edges[0]]],
                                       axis=1)  # (batch_idx, head, rela, tail)
        sampled_edges = torch.LongTensor(sampled_edges).cuda()

        # index to nodes
        head_nodes, head_index = torch.unique(sampled_edges[:, [0, 1]], dim=0, sorted=True, return_inverse=True)
        tail_nodes, tail_index = torch.unique(sampled_edges[:, [0, 3]], dim=0, sorted=True, return_inverse=True)

        mask = sampled_edges[:, 2] == (self.n_rel * 2)
        _, old_idx = head_index[mask].sort()
        old_nodes_new_idx = tail_index[mask][old_idx]

        sampled_edges = torch.cat([sampled_edges, head_index.unsqueeze(1), tail_index.unsqueeze(1)], 1)

        return tail_nodes, sampled_edges, old_nodes_new_idx

    def get_batch(self, batch_idx, steps=2, data='train'):
        if data == 'train':
            return self.tra_train[batch_idx]
        if data == 'valid':
            query, answer = np.array(self.valid_q), np.array(self.valid_a)
            n_ent = self.n_ent
        if data == 'test':
            query, answer = np.array(self.test_q), np.array(self.test_a)
            n_ent = self.n_ent_ind

        subs = []
        rels = []
        objs = []

        subs = query[batch_idx, 0]
        rels = query[batch_idx, 1]
        objs = np.zeros((len(batch_idx), n_ent))
        for i in range(len(batch_idx)):
            objs[i][answer[batch_idx[i]]] = 1
        return subs, rels, objs

    def shuffle_train(self, ):
        rand_idx = np.random.permutation(self.n_train)
        self.tra_train = self.tra_train[rand_idx]

    def get_filter(self, data='valid'):
        filters = defaultdict(lambda: set())
        if data == 'valid':
            for triple in self.tra_train:
                h, r, t = triple
                filters[(h, r)].add(t)
            for triple in self.tra_valid:
                h, r, t = triple
                filters[(h, r)].add(t)
            for triple in self.tra_test:
                h, r, t = triple
                filters[(h, r)].add(t)
        else:
            for triple in self.ind_train:
                h, r, t = triple
                filters[(h, r)].add(t)
            for triple in self.ind_valid:
                h, r, t = triple
                filters[(h, r)].add(t)
            for triple in self.ind_test:
                h, r, t = triple
                filters[(h, r)].add(t)
        return filters


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
        for e in self.entity2id_ind:
            self.id2entity_ind[self.entity2id_ind[e]] = e

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
        G_test = nx.DiGraph()
        with open(os.path.join(ind_dir, 'train.txt')) as f:
            for line in f:
                h, r, t = line.strip().split()
                triplets.append([h, r, t])
                triplets.append([t, r + '_inv', h])
                G_test.add_edge(h, t, relation=r)
                G_test.add_edge(t, h, relation=str(r + '_inv'))
                relations.add(r)
                relations.add(r + '_inv')

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
        accuracy_tensor = torch.zeros([len(relations) + 1, len(relations) + 1], dtype=torch.bool).cuda()
        recall_tensor = torch.zeros([len(relations) + 1, len(relations) + 1], dtype=torch.bool).cuda()
        if accuracy_tensor_indices.shape[0]>0:
            accuracy_tensor[accuracy_tensor_indices[:, 0], accuracy_tensor_indices[:, 1]] = True
        if recall_tensor_indices.shape[0] > 0:
            recall_tensor[recall_tensor_indices[:, 0], recall_tensor_indices[:, 1]] = True

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(os.path.join(save_dir, "distinctive_neighbors.pkl"), "wb") as f:
            torch.save([accuracy_tensor, recall_tensor], f)
        return accuracy_tensor, recall_tensor

    def get_subgraph(self, G, neighbor_filter, center_node, query_relation, depth):
        G_sub = nx.DiGraph()
        filtered_relations = neighbor_filter[query_relation]
        for v in G[center_node]:
            if G[center_node][v]['relation'] in filtered_relations:
                if depth >= 0:
                    G_get = self.get_subgraph(G, neighbor_filter, v, G[center_node][v]['relation'], depth - 1)
                    G_sub = nx.compose(G_sub, G_get)
                G_sub.add_edge(center_node, v, relation=G[center_node][v]['relation'])
                G_sub.add_edge(v, center_node, relation=G[v][center_node]['relation'])
        return G_sub

    def get_complement_subgraph(self, G, neighbor_filter, center_node, query_relation, depth):
        G_sub = nx.DiGraph()
        filtered_relations = neighbor_filter[query_relation]
        for v in G[center_node]:
            if G[center_node][v]['relation'] not in filtered_relations:
                if depth >= 0:
                    G_get = self.get_complement_subgraph(G, neighbor_filter, v, G[center_node][v]['relation'],
                                                         depth - 1)
                    G_sub = nx.compose(G_sub, G_get)
                G_sub.add_edge(center_node, v, relation=G[center_node][v]['relation'])
                G_sub.add_edge(v, center_node, relation=G[v][center_node]['relation'])
        return G_sub

    def graph2triple(self, graph, mode='transductive'):
        triples = []
        if mode == 'transductive':
            for edge in graph.edges:
                triples.append([self.entity2id[edge[0]], self.relation2id[graph[edge[0]][edge[1]]['relation']],
                                self.entity2id[edge[1]]])
        else:
            for edge in graph.edges:
                triples.append([self.entity2id_ind[edge[0]], self.relation2id[graph[edge[0]][edge[1]]['relation']],
                                self.entity2id_ind[edge[1]]])
        triples = np.array(triples)
        return triples

    def get_distinctive_neighbors(self, nodes, query_relations, distinctive_neighbors, head_index_full, tail_index_full,
                                  mode='transductive', complement=False):

        if mode == 'transductive':
            KG = self.tra_KG
            M_sub = self.tra_sub
            n_ent = self.n_ent
        else:
            KG = self.ind_KG
            M_sub = self.ind_sub
            n_ent = self.n_ent_ind
        node_1hot = csr_matrix((np.ones(len(nodes)), (nodes[:, 1], nodes[:, 0])), shape=(n_ent, nodes.shape[0]))
        edge_1hot = M_sub.dot(node_1hot)
        edges = np.nonzero(edge_1hot)
        relations_next = KG[edges[0]][:, 1]

        relations_next=self.inverse_relations(relations_next)
        query_relations_next = query_relations[edges[1]]

        unique_relations, inverse_indices = np.unique(np.stack([relations_next, query_relations_next], axis=1), axis=0,
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
