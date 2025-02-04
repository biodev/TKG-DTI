'''

'''

import argparse
from tkgdti.train.train_gnn import train_gnn
import os
import torch

def get_args():
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, default='../data/HeteroA/processed/FOLD_0', help="path to the data dir")
    parser.add_argument("--out", type=str, default='../output/pathgnn/', help="path to output dir")
    parser.add_argument("--wd", type=float, default=0, help="weight decay")
    parser.add_argument("--channels", type=int, default=8, help="number of hidden channels")
    parser.add_argument("--layers", type=int, default=5, help="number of GNN layers")
    parser.add_argument("--batch_size", type=int, default=5, help="batch size")
    parser.add_argument("--n_epochs", type=int, default=100, help="number of training epochs")
    parser.add_argument("--num_workers", type=int, default=10, help="number of data loader workers")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--dropout", type=float, default=0., help="dropout rate")
    parser.add_argument("--verbose", action='store_true', default=False, help="verbosity")
    parser.add_argument("--nonlin", default='elu', type=str, help="nonlinearity to use in GNN")
    parser.add_argument("--heads", default=1, type=int, help="number of attention heads")
    parser.add_argument("--bias", action='store_true', help="whether to use bias in GNN layers")
    parser.add_argument("--edge_dim", default=2, type=int, help="dimension of edge features")
    parser.add_argument("--checkpoint", action='store_true', default=False, help="whether to use gradient checkpointing")
    parser.add_argument("--log_every", default=1, type=int, help="log every n epochs")
    parser.add_argument("--residual", action='store_true', default=False, help="whether to use residual connections")
    parser.add_argument("--compile", action='store_true', default=False, help="whether to compile the model")
    parser.add_argument("--norm", default='layer', type=str, help="normalization to use in GNN [layer, batch, none]")
    parser.add_argument("--patience", default=5, type=int, help="early stopping patience")
    parser.add_argument("--conv", default='gat', type=str, help="GNN convolution to use")
    parser.add_argument("--target_metric", default='mrr', type=str, help="metric to use for early stopping [hits@10, mrr, auroc]")
    parser.add_argument("--target_relation", default='drug,targets,gene', type=str, help="tuple key of target relation")
    parser.add_argument("--heteroA", action='store_true', default=False, help="whether the data is from the HeteroA dataset")
    parser.add_argument("--remove_relation_idx", default=None, type=int, help="index of relation to remove from knowledge graph")

    args = parser.parse_args()
    args.target_relation = tuple(args.target_relation.split(','))
    return args


if __name__ == '__main__': 

    args = get_args()
    print(args)
    print() 

    os.makedirs(args.out, exist_ok=True)

    config = {'lr': args.lr, 
              'wd': args.wd, 
              'channels': args.channels, 
              'batch_size': args.batch_size, 
              'n_epochs': args.n_epochs, 
              'layers': args.layers,
              'num_workers': args.num_workers,
              'nonlin': args.nonlin,
              'heads': args.heads,
              'bias': args.bias,
              'edge_dim': args.edge_dim,
              'checkpoint': args.checkpoint,
              'dropout':args.dropout,
              'residual': args.residual,
              'conv': args.conv,
              'norm': args.norm,
              'remove_relation_idx':args.remove_relation_idx,
              }
    
    model = train_gnn(config, args)