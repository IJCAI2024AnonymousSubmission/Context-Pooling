import os
import argparse
import torch
import numpy as np
from load_data import DistinctiveDataLoader
from base_model import BaseModel
from utils import select_gpu
from base_model import DistinctiveModel

from ray import tune
import ray
import random

import inspect, re

parser = argparse.ArgumentParser(description="Parser for RED-GNN")
parser.add_argument('--data_path', type=str, default='inductive/data/fb237_v1')
parser.add_argument('--result_dir', type=str, default='inductive/results/fb237_v1')
parser.add_argument('--seed', type=str, default=1234)

args = parser.parse_args()


def varname(p):
  for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
    m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
    if m:
      return m.group(1)

class Options(object):
    pass


cwd = os.getcwd()

def objective(config):
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
        opts.lr = 0.005
        opts.lamb = 0.0002
        opts.decay_rate = 0.991
        opts.hidden_dim = 64
        opts.attn_dim = 5
        opts.dropout = 0.21
        opts.act = 'idd'
        opts.n_layer = 5
        opts.n_batch = 100
    elif dataset == 'fb237_v1':
        opts.decay_rate = 0.994
        opts.attn_dim = 5
        opts.act = 'relu'
        opts.n_layer = 3
        opts.n_batch = 20

        opts.accuracy_threshold = 0.4
        opts.recall_threshold = 0.02
        opts.min_distinctive_depth = 4
        opts.max_distinctive_depth = 6

        opts.lr = 0.0097
        opts.lamb = 0.0085
        opts.hidden_dim = 64
        opts.dropout = 0.21

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

    elif dataset == 'WN18RR_v2':
        opts.lr = 0.0016
        opts.lamb = 0.0004
        opts.decay_rate = 0.994
        opts.hidden_dim = 48
        opts.attn_dim = 3
        opts.dropout = 0.02
        opts.act = 'relu'
        opts.n_layer = 5
        opts.n_batch = 20
    elif dataset == 'fb237_v2':
        opts.lr = 0.0077
        opts.lamb = 0.0002
        opts.decay_rate = 0.993
        opts.hidden_dim = 48
        opts.attn_dim = 5
        opts.dropout = 0.3
        opts.act = 'relu'
        opts.n_layer = 3
        opts.n_batch = 10

        opts.accuracy_threshold = 0.4
        opts.recall_threshold = 0.02
        opts.min_distinctive_depth = 1
        opts.max_distinctive_depth = 4

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
    elif dataset == 'fb237_v3':
        opts.lr = 0.0006
        opts.lamb = 0.000023
        opts.decay_rate = 0.994
        opts.hidden_dim = 48
        opts.attn_dim = 3
        opts.dropout = 0.27
        opts.act = 'relu'
        opts.n_layer = 3
        opts.n_batch = 20
    elif dataset == 'nell_v3':
        opts.lr = 0.0008
        opts.lamb = 0.0004
        opts.decay_rate = 0.995
        opts.hidden_dim = 16
        opts.attn_dim = 3
        opts.dropout = 0.06
        opts.act = 'relu'
        opts.n_layer = 3
        opts.n_batch = 10

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
    elif dataset == 'nell_v4':
        opts.lr = 0.0005
        opts.lamb = 0.000398
        opts.decay_rate = 1
        opts.hidden_dim = 16
        opts.attn_dim = 5
        opts.dropout = 0.1472
        opts.act = 'tanh'
        opts.n_layer = 5
        opts.n_batch = 20

    opts.lr=config['opts.lr']
    opts.lamb=config['opts.lamb']
    opts.hidden_dim=config['opts.hidden_dim']
    opts.dropout=config['opts.dropout']
    opts.accuracy_threshold = config['opts.accuracy_threshold']
    opts.recall_threshold = config['opts.recall_threshold']
    # opts.min_distinctive_depth=config['opts.min_distinctive_depth']
    # opts.max_distinctive_depth=min(config['opts.min_distinctive_depth']+config['opts.depth_range'],8)

    # opts.decay_rate = config['opts.decay_rate']
    # opts.attn_dim = config['opts.attn_dim']
    # opts.act = config['opts.act']
    # opts.n_layer = config['opts.n_layer']
    # opts.n_batch = config['opts.n_batch']

    os.chdir(cwd)

    loader = DistinctiveDataLoader(args.data_path, args.result_dir, True, opts.accuracy_threshold,
                                   opts.recall_threshold,
                                   opts.min_distinctive_depth, opts.max_distinctive_depth)
    opts.n_ent = loader.n_ent
    opts.n_rel = loader.n_rel

    model = DistinctiveModel(opts, loader)
    best_mrr = 0
    best_tmrr=0
    best_str = ''
    best_h1=0
    for epoch in range(10):
        v_mrr, out_str,t_mrr, t_h1, t_h10 = model.train_batch()
        # with open(opts.perf_file, 'a+') as f:
        #     f.write(out_str)

        if v_mrr > best_mrr:
            best_mrr = v_mrr
            best_tmrr=t_mrr
            best_str = out_str
            best_h1=t_h1
            # print(str(epoch) + '\t' + best_str)
    # print(best_str)
    return {'best_tmrr': best_tmrr,'H@1':best_h1,'best_str':best_str,'best_opts':opts}


if __name__ == '__main__':


    search_space = {
        'opts.lr':tune.choice([1e-4 * i for i in range(1, 100)]),
        'opts.lamb':tune.choice([1e-4 * i for i in range(1, 100)]),
        'opts.hidden_dim':tune.choice([32,64,96,128]),
        'opts.dropout': tune.choice([0.01 * i for i in range(10,30)]),
        'opts.accuracy_threshold': tune.choice([0.1*i for i in range(1,7)]),
        'opts.recall_threshold': tune.choice([0.005*i for i in range(1,10)]),
        #
        # 'opts.min_distinctive_depth':tune.choice([i for i in range(1,6)]),
        # 'opts.depth_range':tune.choice([i for i in range(1,5)])

        # 'opts.decay_rate':tune.choice([1-i*0.0001 for i in range(0,51)]),
        # 'opts.attn_dim':tune.choice([i for i in range(2,8)]),
        # 'opts.act':tune.choice(['tanh','idd','relu']),
        # 'opts.n_layer':tune.choice([i for i in range(2,6)]),
        # 'opts.n_batch':tune.choice([i*5 for i in range(1,5)]),
    }
    ray.init()
    results = tune.run(objective, config=search_space, resources_per_trial={"GPU": 0.5, "CPU": 10}, num_samples=40)
    best_trial = results.get_best_trial("best_tmrr", "max", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best test MRR: {best_trial.last_result['best_tmrr']}")
    print(best_trial.last_result['best_str'])
    print(best_trial.last_result['best_opts'].__dict__)
