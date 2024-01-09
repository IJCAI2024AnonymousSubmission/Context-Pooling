import torch
import torch.nn as nn
from torch_scatter import scatter, segment_csr
import copy
import numpy as np


class GNNLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, attn_dim, n_rel, act=lambda x: x):
        super(GNNLayer, self).__init__()
        self.n_rel = n_rel
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.attn_dim = attn_dim
        self.act = act

        self.rela_embed = nn.Embedding(2 * n_rel + 1, in_dim)

        self.Ws_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wr_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wqr_attn = nn.Linear(in_dim, attn_dim)
        self.w_alpha = nn.Linear(attn_dim, 1)

        self.W_h = nn.Linear(in_dim, out_dim, bias=False)


    def forward(self, q_sub, q_rel, hidden, edges, n_node, old_nodes_new_idx):
        # edges:  [batch_idx, head, rela, tail, old_idx, new_idx]
        sub = edges[:, 4]
        rel = edges[:, 2]
        obj = edges[:, 5]



        hs = hidden[sub]
        hr = self.rela_embed(rel)

        r_idx = edges[:, 0]
        h_qr = self.rela_embed(q_rel)[r_idx]

        message = hs + hr
        alpha = torch.sigmoid(self.w_alpha(nn.ReLU()(self.Ws_attn(hs) + self.Wr_attn(hr) + self.Wqr_attn(h_qr))))
        message = alpha * message

        message_agg = scatter(message, index=obj, dim=0, dim_size=n_node, reduce='sum')

        hidden_new = self.act(self.W_h(message_agg))

        return hidden_new


class Deterministic_GNNLayer(GNNLayer):

    # This class is defined to ensure the reproducibility of code.
    # Scatter function is not deterministic, so we rewrite this part with segment_csr.
    def forward(self, q_sub, q_rel, hidden, edges, n_node, old_nodes_new_idx):
        sub = edges[:, 4]
        rel = edges[:, 2]
        obj = edges[:, 5]

        hs = hidden[sub]
        hr = self.rela_embed(rel)

        r_idx = edges[:, 0]
        h_qr = self.rela_embed(q_rel)[r_idx]

        message = hs + hr
        alpha = torch.sigmoid(self.w_alpha(nn.ReLU()(self.Ws_attn(hs) + self.Wr_attn(hr) + self.Wqr_attn(h_qr))))
        message = alpha * message


        sorted_obj, order = torch.sort(obj)

        idx = torch.where(sorted_obj[1:] - sorted_obj[:-1] >= 1)[0] + 1
        idx = torch.cat([torch.tensor([0]).cuda(), idx, torch.tensor([len(sorted_obj)]).cuda()])

        message_agg = segment_csr(message[order], idx)
        padded=torch.zeros(n_node,self.in_dim).cuda()
        if len(sorted_obj)>0:
            padded[sorted_obj[idx[:-1]]]=message_agg

        # message_agg = scatter(message, index=obj, dim=0, dim_size=n_node, reduce='sum')
        # assert len(torch.where(scatter(message, index=obj, dim=0, dim_size=n_node, reduce='sum')-padded!=0))>0
        hidden_new = self.act(self.W_h(padded))

        return hidden_new


class RED_GNN_induc(torch.nn.Module):
    def __init__(self, params, loader):
        super(RED_GNN_induc, self).__init__()
        self.n_layer = params.n_layer
        self.hidden_dim = params.hidden_dim
        self.attn_dim = params.attn_dim
        self.n_rel = params.n_rel
        self.loader = loader
        acts = {'relu': nn.ReLU(), 'tanh': torch.tanh, 'idd': lambda x: x}
        act = acts[params.act]

        self.gnn_layers = []
        for i in range(self.n_layer):
            self.gnn_layers.append(GNNLayer(self.hidden_dim, self.hidden_dim, self.attn_dim, self.n_rel, act=act))
        self.gnn_layers = nn.ModuleList(self.gnn_layers)

        self.dropout = nn.Dropout(params.dropout)
        self.W_final = nn.Linear(self.hidden_dim, 1, bias=False)  # get score
        self.gate = nn.GRU(self.hidden_dim, self.hidden_dim)

    def forward(self, subs, rels, mode='transductive'):
        n = len(subs)

        n_ent = self.loader.n_ent if mode == 'transductive' else self.loader.n_ent_ind

        q_sub = torch.LongTensor(subs).cuda()
        q_rel = torch.LongTensor(rels).cuda()

        h0 = torch.zeros((1, n, self.hidden_dim)).cuda()
        nodes = torch.cat([torch.arange(n).unsqueeze(1).cuda(), q_sub.unsqueeze(1)], 1)
        hidden = torch.zeros(n, self.hidden_dim).cuda()

        for i in range(self.n_layer):
            nodes, edges, old_nodes_new_idx = self.loader.get_neighbors(nodes.data.cpu().numpy(), mode=mode)

            hidden = self.gnn_layers[i](q_sub, q_rel, hidden, edges, nodes.size(0), old_nodes_new_idx)
            h0 = torch.zeros(1, nodes.size(0), hidden.size(1)).cuda().index_copy_(1, old_nodes_new_idx, h0)
            hidden = self.dropout(hidden)
            hidden, h0 = self.gate(hidden.unsqueeze(0), h0)
            hidden = hidden.squeeze(0)

        scores = self.W_final(hidden).squeeze(-1)
        scores_all = torch.zeros((n, n_ent)).cuda()
        scores_all[[nodes[:, 0], nodes[:, 1]]] = scores
        return scores_all


class Distinct_RED_GNN_induc(RED_GNN_induc):
    def __init__(self, params, loader):
        super().__init__(params, loader)
        acts = {'relu': nn.ReLU(), 'tanh': torch.tanh, 'idd': lambda x: x}
        act = acts[params.act]
        self.graphs=[params.accuracy_graph,params.recall_graph,params.accuracy_graph_complement,params.recall_graph_complement]
        self.graph_num=np.sum(np.array(self.graphs))+1
        self.gnn_layers = nn.ModuleList(
            [Deterministic_GNNLayer(self.hidden_dim, self.hidden_dim, self.attn_dim, self.n_rel, act=act) for i in
             range(self.n_layer)])
        if self.graphs[0]:
            self.accuracy_layer = nn.ModuleList(
                [Deterministic_GNNLayer(self.hidden_dim, self.hidden_dim, self.attn_dim, self.n_rel, act=act) for i in
                 range(self.n_layer)])


        if self.graphs[1]:
            self.recall_layer = nn.ModuleList(
                [Deterministic_GNNLayer(self.hidden_dim, self.hidden_dim, self.attn_dim, self.n_rel, act=act) for i in
                 range(self.n_layer)])

        if self.graphs[2]:
            self.accuracy_complement_layer = nn.ModuleList(
                [Deterministic_GNNLayer(self.hidden_dim, self.hidden_dim, self.attn_dim, self.n_rel, act=act) for i
                 in
                 range(self.n_layer)])

        if self.graphs[3]:
            self.recall_complement_layer = nn.ModuleList(
                [Deterministic_GNNLayer(self.hidden_dim, self.hidden_dim, self.attn_dim, self.n_rel, act=act) for i in
                 range(self.n_layer)])
        
        # self.maxpool = nn.MaxPool1d(kernel_size=5)
        # self.conv = nn.Conv2d(in_channels=5, out_channels=1, kernel_size=(1, 1))
        self.W_final = nn.Linear(self.hidden_dim, 1, bias=True)  # get score
        self.gate = nn.GRU(self.hidden_dim*self.graph_num, self.hidden_dim)

    def forward(self, subs, rels, mode='transductive'):
        n = len(subs)

        n_ent = self.loader.n_ent if mode == 'transductive' else self.loader.n_ent_ind

        q_sub = torch.LongTensor(subs).cuda()
        q_rel = torch.LongTensor(rels).cuda()

        h0 = torch.zeros((1, n, self.hidden_dim)).cuda()
        nodes_full_graph = torch.cat([torch.arange(n).unsqueeze(1).cuda(), q_sub.unsqueeze(1)], 1)
        nodes_accuracy = torch.clone(nodes_full_graph)
        nodes_accuracy_complement= torch.clone(nodes_full_graph)
        nodes_recall = torch.clone(nodes_full_graph)
        nodes_recall_complement = torch.clone(nodes_full_graph)
        hidden = torch.zeros(n, self.hidden_dim).cuda()

        inverse_rels=self.loader.inverse_relations(rels)
        query_relations_accuracy=inverse_rels.copy()
        query_relations_accuracy_complement = inverse_rels.copy()
        query_relations_recall = inverse_rels.copy()
        query_relations_recall_complement=inverse_rels.copy()
        for i in range(self.n_layer):
            hidden_new=[]
            nodes_full_graph, edges_full_graph, old_nodes_new_idx_full_graph = self.loader.get_neighbors(
                nodes_full_graph.data.cpu().numpy(), mode=mode)
            hidden_full_graph = self.gnn_layers[i](q_sub, q_rel, hidden, edges_full_graph, nodes_full_graph.size(0),
                                                   old_nodes_new_idx_full_graph)
            hidden_new.append(hidden_full_graph)

            head_indexes = torch.zeros([n, n_ent], dtype=torch.long).cuda()
            head_indexes[edges_full_graph[:, 0], edges_full_graph[:, 1]] = edges_full_graph[:, 4]
            tail_indexes = torch.zeros(n, n_ent, dtype=torch.long).cuda()
            tail_indexes[edges_full_graph[:, 0], edges_full_graph[:, 3]] = edges_full_graph[:, 5]
            # head_indexes = {tuple(edges_full_graph[i][[0, 1]].cpu().tolist()): edges_full_graph[i][4].item() for i in range(len(edges_full_graph))}
            # tail_indexes = {tuple(edges_full_graph[i][[0, 3]].cpu().tolist()): edges_full_graph[i][5].item() for i in range(len(edges_full_graph))}
            if self.graphs[0]:
                nodes_accuracy, edges_accuracy, old_nodes_new_idx_accuracy,query_relations_accuracy = self.loader.get_distinctive_neighbors(
                    nodes_accuracy.data.cpu().numpy(), query_relations_accuracy, self.loader.accuracy_tensor,
                    head_indexes, tail_indexes,
                    mode=mode,complement=False)
                hidden_accuracy = self.accuracy_layer[i](q_sub, q_rel, hidden, edges_accuracy,
                                                         nodes_full_graph.size(0),
                                                         old_nodes_new_idx_accuracy)
                hidden_new.append(hidden_accuracy)

            if self.graphs[1]:
                nodes_recall, edges_recall, old_nodes_new_idx_recall,query_relations_recall = self.loader.get_distinctive_neighbors(
                    nodes_recall.data.cpu().numpy(), query_relations_recall, self.loader.recall_tensor,  head_indexes,
                    tail_indexes,
                    mode=mode,complement=False)
                hidden_recall = self.recall_layer[i](q_sub, q_rel, hidden, edges_recall,
                                                     nodes_full_graph.size(0),
                                                     old_nodes_new_idx_recall)
                hidden_new.append(hidden_recall)

            if self.graphs[2]:
                nodes_accuracy_complement, edges_accuracy_complement, old_nodes_new_idx_accuracy_complement, query_relations_accuracy_complement = self.loader.get_distinctive_neighbors(
                    nodes_accuracy_complement.data.cpu().numpy(), query_relations_accuracy_complement, self.loader.accuracy_tensor,
                    head_indexes, tail_indexes,
                    mode=mode, complement=True)
                hidden_accuracy_complement = self.accuracy_complement_layer[i](q_sub, q_rel, hidden, edges_accuracy_complement,
                                                         nodes_full_graph.size(0),
                                                         old_nodes_new_idx_accuracy_complement)
                hidden_new.append(hidden_accuracy_complement)

            if self.graphs[3]:
                nodes_recall_complement, edges_recall_complement, old_nodes_new_idx_recall_complement, query_relations_recall_complement = self.loader.get_distinctive_neighbors(
                    nodes_recall_complement.data.cpu().numpy(), query_relations_recall_complement, self.loader.recall_tensor, head_indexes,
                    tail_indexes,
                    mode=mode, complement=True)
                hidden_recall_complement = self.recall_complement_layer[i](q_sub, q_rel, hidden, edges_recall_complement,
                                                     nodes_full_graph.size(0),
                                                     old_nodes_new_idx_recall_complement)
                hidden_new.append(hidden_recall_complement)

            hidden = torch.cat(hidden_new,dim=1)

            # hidden = self.maxpool(hidden.permute(1, 2, 0)).squeeze(2)
            # hidden = self.conv(hidden).squeeze(0)

            h0 = torch.zeros(1, nodes_full_graph.size(0), self.hidden_dim).cuda().index_copy_(1,
                                                                                             old_nodes_new_idx_full_graph,
                                                                                             h0)
            hidden = self.dropout(hidden)
            hidden, h0 = self.gate(hidden.unsqueeze(0), h0)
            hidden = hidden.squeeze(0)

        scores = self.W_final(hidden).squeeze(-1)
        scores_all = torch.zeros((n, n_ent)).cuda()
        scores_all[[nodes_full_graph[:, 0], nodes_full_graph[:, 1]]] = scores
        return scores_all
