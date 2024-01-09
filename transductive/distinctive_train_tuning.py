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

from distinctive_train import objective







class Options(object):
    pass



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parser for RED-GNN")
    parser.add_argument('--data_path', type=str, default='data/new/WN18RR_v4')
    parser.add_argument('--result_dir', type=str, default='results/new/WN18RR_v4')
    parser.add_argument('--seed', type=str, default=1234)

    args = parser.parse_args()
    cwd = os.getcwd()
    search_space = {


        # 'opts.lr':tune.choice([1e-5 * i for i in range(1, 1000)]),
        # 'opts.lamb':tune.choice([1e-4 * i for i in range(1, 100)]),
        # 'opts.hidden_dim':tune.choice([32,64,96,128]),
        # 'opts.dropout': tune.choice([0.01 * i for i in range(10,30)]),
        # 'opts.decay_rate':tune.choice([1-i*0.0001 for i in range(0,51)]),


        # 'opts.attn_dim':tune.choice([i for i in range(2,8)]),
        # 'opts.act':tune.choice(['tanh','idd','relu']),
        # 'opts.n_layer':tune.choice([i for i in range(2,6)]),
        # 'opts.n_batch':tune.choice([i*5 for i in range(1,5)]),

        'opts.accuracy_threshold': tune.choice([0.01*i for i in range(1,90)]),
        'opts.recall_threshold': tune.choice([0.01*i for i in range(1,90)]),
        'opts.accuracy_graph':tune.choice([True,False]),
        'opts.recall_graph':tune.choice([True,False]),
        'opts.accuracy_graph_complement':tune.choice([True,False]),
        'opts.recall_graph_complement':tune.choice([True,False]),





        'epochs':50,
        'tuning': True,
        'cwd': cwd,
        'args': args
    }
    ray.init()
    results = tune.run(objective, config=search_space, resources_per_trial={"GPU": 2/2, "CPU": 40/2}, num_samples=40)
    best_trial = results.get_best_trial("best_tmrr", "max", "last")
    print(args)
    print(f"Best trial config: {best_trial.config}")
    print(f"Best test MRR: {best_trial.last_result['best_tmrr']}")
    print(best_trial.last_result['best_str'])
    print(best_trial.last_result['best_opts'])
