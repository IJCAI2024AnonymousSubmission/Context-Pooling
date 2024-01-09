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
    opts.perf_file = os.path.join(results_dir, dataset + '_perf.txt')

    # gpu = select_gpu()
    # torch.cuda.set_device(gpu)
    # print('gpu:', gpu)
    torch.cuda.set_device('cuda:0')

    if dataset == 'WN18RR_v1':
        opts.lr = 0.0062
        opts.lamb = 0.0098
        opts.dropout = 0.29
        opts.hidden_dim = 32

        opts.decay_rate = 0.9986
        opts.attn_dim = 4

        opts.act = 'tanh'
        opts.n_layer = 3
        opts.n_batch = 5

        opts.accuracy_threshold = 0.1
        opts.recall_threshold = 0.03

        opts.accuracy_graph = True
        opts.recall_graph = True
        opts.accuracy_graph_complement = True
        opts.recall_graph_complement = True

        opts.epochs = 50

    elif dataset == 'fb237_v1':
        opts.decay_rate = 0.9978
        opts.attn_dim = 6
        opts.act = 'tanh'
        opts.n_layer = 3
        opts.n_batch = 5

        opts.lr = 0.0096
        opts.lamb = 0.0078
        opts.hidden_dim = 128
        opts.dropout = 0.22

        opts.accuracy_threshold = 0.4
        opts.recall_threshold = 0.015

        opts.accuracy_graph = True
        opts.recall_graph = True
        opts.accuracy_graph_complement = True
        opts.recall_graph_complement = True

        opts.epochs = 50

    elif dataset == 'nell_v1':
        opts.lr = 0.0021
        opts.lamb = 0.000189
        opts.decay_rate = 0.9937
        opts.hidden_dim = 48
        opts.attn_dim = 5
        opts.dropout = 0.2460
        opts.act = 'relu'
        opts.n_layer = 5
        opts.n_batch = 10

        opts.accuracy_threshold = 0.86
        opts.recall_threshold = 0.89

        opts.accuracy_graph = True
        opts.recall_graph = True
        opts.accuracy_graph_complement = False
        opts.recall_graph_complement = True

        opts.epochs = 10

    elif dataset == 'WN18RR_v2':
        opts.lr = 0.00195
        opts.lamb = 0.0004
        opts.decay_rate = 0.994
        opts.hidden_dim = 48
        opts.attn_dim = 3
        opts.dropout = 0.02
        opts.act = 'relu'
        opts.n_layer = 5
        opts.n_batch = 20

        opts.accuracy_threshold = 0.46
        opts.recall_threshold = 0.45

        opts.accuracy_graph = True
        opts.recall_graph = False
        opts.accuracy_graph_complement = True
        opts.recall_graph_complement = True

        opts.epochs = 5

    elif dataset == 'fb237_v2':

        # opts.lr = 0.0075
        # opts.lamb = 0.0067
        # opts.hidden_dim = 32
        # opts.dropout = 0.18
        # opts.accuracy_threshold = 0.2
        # opts.recall_threshold = 0.03
        #
        #
        # opts.decay_rate = 0.997
        # opts.attn_dim = 3
        # opts.act = 'relu'
        # opts.n_layer = 4
        # opts.n_batch = 5

        opts.lr = 0.00581
        opts.lamb = 0.0002
        opts.decay_rate = 0.993
        opts.hidden_dim = 48
        opts.attn_dim = 5
        opts.dropout = 0.3
        opts.act = 'relu'
        opts.n_layer = 3
        opts.n_batch = 10

        opts.accuracy_threshold = 0.49
        opts.recall_threshold = 0.22
        opts.accuracy_graph = True
        opts.recall_graph = True
        opts.accuracy_graph_complement = True
        opts.recall_graph_complement = False

        opts.epochs = 5

    elif dataset == 'nell_v2':
        opts.lr = 0.0075
        opts.lamb = 0.000066
        opts.decay_rate = 0.9996
        opts.hidden_dim = 48
        opts.attn_dim = 5
        opts.dropout = 0.2881
        opts.act = 'relu'
        opts.n_layer = 3
        opts.n_batch = 100

        opts.accuracy_threshold = 0.85
        opts.recall_threshold = 0.84

        opts.accuracy_graph = False
        opts.recall_graph = True
        opts.accuracy_graph_complement = False
        opts.recall_graph_complement = True

        opts.epochs = 10

    elif dataset == 'WN18RR_v3':
        opts.lr = 0.0014
        opts.lamb = 0.000034
        opts.decay_rate = 0.991
        opts.hidden_dim = 64
        opts.attn_dim = 5
        opts.dropout = 0.28
        opts.act = 'tanh'
        opts.n_layer = 5
        opts.n_batch = 20

        opts.accuracy_threshold = 0.38
        opts.recall_threshold = 0.19
        opts.accuracy_graph = True
        opts.recall_graph = False
        opts.accuracy_graph_complement = False
        opts.recall_graph_complement = False

        opts.epochs = 20

    elif dataset == 'fb237_v3':
        opts.lr = 0.0006
        opts.lamb = 0.000023
        opts.decay_rate = 0.994
        opts.hidden_dim = 48

        opts.attn_dim = 7
        opts.dropout = 0.27
        opts.act = 'relu'
        opts.n_layer = 4
        opts.n_batch = 5

        opts.accuracy_threshold = 0.18
        opts.recall_threshold = 0.55

        opts.accuracy_graph = True
        opts.recall_graph = True
        opts.accuracy_graph_complement = False
        opts.recall_graph_complement = True

        opts.epochs = 10

    elif dataset == 'nell_v3':
        opts.lr = 0.0008
        opts.lamb = 0.0004
        opts.decay_rate = 0.995
        opts.hidden_dim = 16
        opts.dropout = 0.06

        opts.attn_dim = 7
        opts.act = 'relu'
        opts.n_layer = 3
        opts.n_batch = 15

        opts.accuracy_threshold = 0.78
        opts.recall_threshold = 0.8

        opts.accuracy_graph = True
        opts.recall_graph = True
        opts.accuracy_graph_complement = True
        opts.recall_graph_complement = True

        opts.epochs = 10

    elif dataset == 'WN18RR_v4':
        opts.lr = 0.006
        opts.lamb = 0.000132
        opts.decay_rate = 0.991
        opts.hidden_dim = 32
        opts.attn_dim = 5
        opts.dropout = 0.11
        opts.act = 'relu'
        opts.n_layer = 5
        opts.n_batch = 10

        opts.accuracy_threshold = 0.14
        opts.recall_threshold = 0.22

        opts.accuracy_graph = True
        opts.recall_graph = False
        opts.accuracy_graph_complement = False
        opts.recall_graph_complement = True

        opts.epochs = 10

    elif dataset == 'fb237_v4':
        opts.lr = 0.0052
        opts.lamb = 0.000018
        opts.decay_rate = 0.999
        opts.hidden_dim = 48
        opts.attn_dim = 5
        opts.dropout = 0.07
        opts.act = 'idd'
        opts.n_layer = 5
        opts.n_batch = 20

        opts.accuracy_threshold = 0.54
        opts.recall_threshold = 0.73

        opts.accuracy_graph = False
        opts.recall_graph = True
        opts.accuracy_graph_complement = False
        opts.recall_graph_complement = False

        opts.epochs = 10

    elif dataset == 'nell_v4':
        opts.lr = 0.005
        opts.lamb = 0.000398
        opts.hidden_dim = 16
        opts.dropout = 0.1472
        opts.decay_rate = 1

        opts.attn_dim = 4
        opts.act = 'relu'
        opts.n_layer = 3
        opts.n_batch = 20

        opts.accuracy_threshold = 0.11
        opts.recall_threshold = 0.37

        opts.accuracy_graph = False
        opts.recall_graph = True
        opts.accuracy_graph_complement = True
        opts.recall_graph_complement = False

        opts.epochs = 8



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
    best_tmrr=0
    best_str = ''
    best_h1=0
    for epoch in range(opts.epochs):
        v_mrr, out_str,t_mrr, t_h1, t_h10 = model.train_batch()
        # with open(opts.perf_file, 'a+') as f:
        #     f.write(out_str)

        if v_mrr > best_mrr:
            best_mrr = v_mrr
            best_tmrr=t_mrr
            best_str = out_str
            best_h1=t_h1
            if config['tuning'] is False:
                print(str(epoch) + '\t' + best_str)
    # print(best_str)
    return {'best_tmrr': best_tmrr,'H@1':best_h1,'best_str':best_str,'best_opts':opts.__dict__}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parser for RED-GNN")
    parser.add_argument('--data_path', type=str, default='data/fb237_v3')
    parser.add_argument('--result_dir', type=str, default='results/fb237_v3')
    parser.add_argument('--seed', type=str, default=1234)

    args = parser.parse_args()
    print(args)
    results=objective({'tuning': False, 'args': args,'epochs': 50})
    print(results['best_opts'])
