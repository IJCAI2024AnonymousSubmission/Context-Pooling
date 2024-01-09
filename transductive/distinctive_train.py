import os
import argparse
import torch
import numpy as np
from load_data import DistinctiveDataLoader
from base_model import BaseModel
from utils import select_gpu
from base_model import DistinctiveModel
import random

from ray import tune


class Options(object):
    pass


def objective(config):
    if config['tuning'] is True:
        os.chdir(config['cwd'])
    args = config['args']
    torch.use_deterministic_algorithms(True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = args.data_path
    dataset = dataset.rstrip('\t\n').split('/')[-1]

    results_dir = args.result_dir
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    opts = Options
    # opts.perf_file = os.path.join(results_dir,  dataset + '_perf.txt')

    # gpu = select_gpu()
    # torch.cuda.set_device(gpu)
    # print('gpu:', gpu)
    torch.cuda.set_device('cuda:0')
    if dataset == 'fb237_v1':
        opts.lr = 0.0009
        opts.decay_rate = 0.9938
        opts.lamb = 0.000080
        opts.hidden_dim = 48
        opts.attn_dim = 5
        opts.n_layer = 4
        opts.dropout = 0.0391
        opts.act = 'relu'
        opts.n_batch = 5
        opts.n_tbatch = 1

        opts.epochs = 50
        opts.amp = False
    elif dataset == 'fb237_v2':
        opts.lr = 0.0009
        opts.decay_rate = 0.9938
        opts.lamb = 0.000080
        opts.hidden_dim = 48
        opts.attn_dim = 5
        opts.n_layer = 4
        opts.dropout = 0.0391
        opts.act = 'relu'
        opts.n_batch = 5
        opts.n_tbatch = 1

        opts.epochs = 50
        opts.amp = False
    elif dataset == 'fb237_v3':
        opts.lr = 0.0009
        opts.decay_rate = 0.9938
        opts.lamb = 0.000080
        opts.hidden_dim = 48
        opts.attn_dim = 5
        opts.n_layer = 4
        opts.dropout = 0.0391
        opts.act = 'relu'
        opts.n_batch = 5
        opts.n_tbatch = 1

        opts.epochs = 50
        opts.amp = False
    elif dataset == 'fb237_v4':
        opts.lr = 0.0009
        opts.decay_rate = 0.9938
        opts.lamb = 0.000080
        opts.hidden_dim = 48
        opts.attn_dim = 5
        opts.n_layer = 3
        opts.dropout = 0.0391
        opts.act = 'relu'
        opts.n_batch = 5
        opts.n_tbatch = 1

        opts.accuracy_threshold = 0.1
        opts.recall_threshold = 0.59

        opts.accuracy_graph = False
        opts.recall_graph = True
        opts.accuracy_graph_complement = True
        opts.recall_graph_complement = False

        opts.epochs = 50
        opts.amp = False
    elif dataset == 'nell_v1':
        opts.lr = 0.0011
        opts.decay_rate = 0.9938
        opts.lamb = 0.000089
        opts.hidden_dim = 48
        opts.attn_dim = 5
        opts.n_layer = 5
        opts.dropout = 0.2593
        opts.act = 'relu'
        opts.n_batch = 5
        opts.n_tbatch = 1

        opts.epochs = 50
        opts.amp = False
    elif dataset == 'nell_v2':
        opts.lr = 0.0011
        opts.decay_rate = 0.9938
        opts.lamb = 0.000089
        opts.hidden_dim = 48
        opts.attn_dim = 5
        opts.n_layer = 5
        opts.dropout = 0.2593
        opts.act = 'relu'
        opts.n_batch = 5
        opts.n_tbatch = 1

        opts.epochs = 50
        opts.amp = False
    elif dataset == 'nell_v3':
        opts.lr = 0.0011
        opts.decay_rate = 0.9938
        opts.lamb = 0.000089
        opts.hidden_dim = 48
        opts.attn_dim = 5
        opts.n_layer = 5
        opts.dropout = 0.2593
        opts.act = 'relu'
        opts.n_batch = 5
        opts.n_tbatch = 1

        opts.epochs = 50
        opts.amp = False
    elif dataset == 'nell_v4':
        opts.lr = 0.0011
        opts.decay_rate = 0.9938
        opts.lamb = 0.000089
        opts.hidden_dim = 48
        opts.attn_dim = 5
        opts.n_layer = 5
        opts.dropout = 0.2593
        opts.act = 'relu'
        opts.n_batch = 5
        opts.n_tbatch = 1

        opts.epochs = 50
        opts.amp = False
    elif dataset == 'WN18RR_v1':
        opts.lr = 0.0003
        opts.decay_rate = 0.994
        opts.lamb = 0.00014
        opts.hidden_dim = 64
        opts.attn_dim = 5
        opts.n_layer = 5
        opts.dropout = 0.02
        opts.act = 'idd'
        opts.n_batch = 50
        opts.n_tbatch = 50

        opts.epochs = 50
        opts.amp = False
    elif dataset == 'WN18RR_v2':
        opts.lr = 0.0003
        opts.decay_rate = 0.994
        opts.lamb = 0.00014
        opts.hidden_dim = 64
        opts.attn_dim = 5
        opts.n_layer = 5
        opts.dropout = 0.02
        opts.act = 'idd'
        opts.n_batch = 50
        opts.n_tbatch = 50

        opts.epochs = 50
        opts.amp = False
    elif dataset == 'WN18RR_v3':
        opts.lr = 0.0003
        opts.decay_rate = 0.994
        opts.lamb = 0.00014
        opts.hidden_dim = 64
        opts.attn_dim = 5
        opts.n_layer = 5
        opts.dropout = 0.02
        opts.act = 'idd'
        opts.n_batch = 50
        opts.n_tbatch = 50

        opts.epochs = 50
        opts.amp = False
    elif dataset == 'WN18RR_v4':
        opts.lr = 0.0003
        opts.decay_rate = 0.994
        opts.lamb = 0.00014
        opts.hidden_dim = 64
        opts.attn_dim = 5
        opts.n_layer = 5
        opts.dropout = 0.02
        opts.act = 'idd'
        opts.n_batch = 50
        opts.n_tbatch = 50

        opts.epochs = 50
        opts.amp = False

    if config['tuning'] is True:
        opts.epochs = config['epochs']

        # opts.lr=config['opts.lr']
        # opts.lamb=config['opts.lamb']
        # opts.hidden_dim=config['opts.hidden_dim']
        # opts.dropout=config['opts.dropout']
        # opts.decay_rate = config['opts.decay_rate']

        # opts.attn_dim = config['opts.attn_dim']
        # opts.act = config['opts.act']
        # opts.n_layer = config['opts.n_layer']
        # opts.n_batch = config['opts.n_batch']

        opts.accuracy_threshold = config['opts.accuracy_threshold']
        opts.recall_threshold = config['opts.recall_threshold']
        opts.accuracy_graph = config['opts.accuracy_graph']
        opts.recall_graph = config['opts.recall_graph']
        opts.accuracy_graph_complement = config['opts.accuracy_graph_complement']
        opts.recall_graph_complement = config['opts.recall_graph_complement']

    loader = DistinctiveDataLoader(args.data_path, args.result_dir, True, opts.accuracy_threshold,
                                   opts.recall_threshold)
    opts.n_ent = loader.n_ent
    opts.n_rel = loader.n_rel

    model = DistinctiveModel(opts, loader)

    best_mrr = 0
    best_tmrr = 0
    best_str = ''
    best_h1 = 0
    for epoch in range(opts.epochs):
        v_mrr, out_str, t_mrr, t_h1, t_h10 = model.train_batch()
        # with open(opts.perf_file, 'a+') as f:
        #     f.write(out_str)

        if v_mrr > best_mrr:
            best_mrr = v_mrr
            best_tmrr = t_mrr
            best_str = out_str
            best_h1 = t_h1
            if config['tuning'] is False:
                print(str(epoch) + '\t' + best_str)
    # print(best_str)
    return {'best_tmrr': best_tmrr, 'H@1': best_h1, 'best_str': best_str, 'best_opts': opts.__dict__}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parser for RED-GNN")
    parser.add_argument('--data_path', type=str, default='data/new/fb237_v4')
    parser.add_argument('--result_dir', type=str, default='results/fb237_v4')
    parser.add_argument('--seed', type=str, default=1234)

    args = parser.parse_args()
    print(args)
    results = objective({'tuning': False, 'args': args, 'epochs': 20})
    print(results['best_opts'])
