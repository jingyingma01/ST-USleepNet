import os
import argparse
import random
import time
import torch
import wandb
import numpy as np
from network import GNet
from trainer import Trainer
from utils.data_loader import FileLoader
import warnings
warnings.filterwarnings('ignore')

def get_args():
    parser = argparse.ArgumentParser(description='Args for graph predition')
    parser.add_argument('--cuda', default = 0, type = int, help = 'CUDA device number')
    parser.add_argument('--seed', type = int, default = 1, help = 'seed')
    parser.add_argument('--data', default = 'ISRUC_S3', help = 'data folder name')
    parser.add_argument('--num_node', type = int, default = 10, help = 'num of channels')
    parser.add_argument('--num_patch', type = int, default = 5, help='Number of Patches')
    parser.add_argument('--feat_dim', type = int, default = 600, help='Feature Dim')
    parser.add_argument('--wdb', type = bool, default = False)
    parser.add_argument('--sweep', type = bool, default = False)
    parser.add_argument('--fold', type = int, default = 0, help = 'fold (0...9)')
    parser.add_argument('--num_epochs', type = int, default = 40, help = 'epochs')
    parser.add_argument('--batch', type = int, default = 64, help = 'batch size')
    parser.add_argument('--lr', type = float, default = 0.001, help = 'learning rate')
    parser.add_argument('--drop_n', type = float, default = 0.2, help = 'drop net')
    parser.add_argument('--drop_c', type = float, default = 0.4, help = 'drop output')
    parser.add_argument('--act_n', type = str, default = 'ELU', help = 'network act')
    parser.add_argument('--act_c', type = str, default = 'ELU', help = 'output act')
    parser.add_argument('--gcn_h', nargs = '+', default = '1024 512 256 128 128', help = 'GCN hidden layer')
    parser.add_argument('--l_n', type = int, default = 4, help = 'The layer of Unet')
    parser.add_argument('--ks', nargs = '+', default = '0.9 0.8 0.7 0.6')
    parser.add_argument('--cs', nargs = '+', default = '0.5 0.5 0.5 0.2')
    parser.add_argument('--sch', type = int, default = 2, help = 'scheduler')
    parser.add_argument('--chs', nargs = '+', default = '32 64 128 256 512', help = 'CNN Channels')
    parser.add_argument('--kernal', nargs = '+', default = '15 9 7 3 3', help = 'kernal')
    parser.add_argument('--delta_t', type = float, default = 0.8, help='Adjacency Time Matrix')
    parser.add_argument('--delta_p', type = float, default = 0.9, help='Adjacency Position Matrix')
    parser.add_argument('--num_class', type = int, default = 5, help = 'Number of Classification')
    parser.add_argument('--weightDecay', type = float, default = 0.005)
    parser.add_argument('--lrStepSize', type = int, default = 10)
    parser.add_argument('--lrGamma', type = int, default = 0.1)
    parser.add_argument('--lrFactor', type = float, default = 0.5)
    parser.add_argument('--lrPatience', type = int, default = 5)
    args, _ = parser.parse_known_args()
    return args


def set_random(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def app_run(args, config, G_data):
    net = GNet(config)
    trainer = Trainer(args, net, G_data)
    trainer.train()

def main():
    args = get_args()
    if args.sweep:
        wandb.init(
            project = "wandb project name"
        )
        config = wandb.config
    elif args.wdb:
        wandb.init(
            project = "wandb project name",
            config = args
        )
        config = args
    else:
        config = args
    print(config)
    set_random(config.seed)
    G_data = FileLoader(config).load_data()
    print('start training ------> fold', config.fold)
    app_run(args, config, G_data)

if __name__ == "__main__":
    main()