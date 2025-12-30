'''

'''

import argparse
from tkgdti.train.train_gekc import train_gekc

def get_args():
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, default='../data/HeteroA/processed/FOLD_0', help="path to the data dir")
    parser.add_argument("--out", type=str, default='../output/', help="path to output dir")
    parser.add_argument("--optim", type=str, default='adam', help="optimizer [adam, adagrad]")
    parser.add_argument("--wd", type=float, default=0, help="weight decay")
    parser.add_argument("--channels", type=int, default=1024, help="number of hidden channels")
    parser.add_argument("--batch_size", type=int, default=25000, help="batch size")
    parser.add_argument("--n_epochs", type=int, default=100, help="number of training epochs")
    parser.add_argument("--num_workers", type=int, default=20, help="number of data loader workers")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--dropout", type=float, default=0., help="dropout rate")
    parser.add_argument("--lr_scheduler", action='store_true', default=False, help="whether to use lr scheduler (reduce on plateu)")
    parser.add_argument("--verbose", action='store_true', default=False, help="verbosity")
    parser.add_argument("--log_every", default=5, type=int, help="how often to log validation performance during training")
    parser.add_argument("--target_relations", default=None, nargs='*', type=int, help="target relations to use for loss scaling")
    parser.add_argument("--patience", default=3, type=int, help="early stopping patience; log_every*patience epochs without improvement")
    parser.add_argument("--use_cpu", action='store_true', default=False, help="use cpu instead of gpu")
    parser.add_argument("--target_relation", default='drug,targets,gene', type=str, help="tuple key of target relation")
    parser.add_argument("--target_metric", default='mrr', type=str, help="metric to use for early stopping [hits@10, mrr, auroc]")
    parser.add_argument("--remove_relation_idx", default=None, type=int, help="index of relation to remove from knowledge graph")
    parser.add_argument("--eval_method", default='all', type=str, choices=['all', 'negatives'], help="evaluation method: 'all' or 'negatives'")

    args = parser.parse_args()
    args.target_relation = tuple(args.target_relation.split(','))
    return args


if __name__ == '__main__': 

    args = get_args()
    print(args)
    print() 

    config = {'model':'complex2',
              'optim':args.optim,
              'lr': args.lr, 
              'wd': args.wd, 
              'channels': args.channels, 
              'batch_size': args.batch_size, 
              'n_epochs': args.n_epochs, 
              'lr_scheduler' : args.lr_scheduler, 
              'dropout':args.dropout,
              'remove_relation_idx':args.remove_relation_idx,}
    
    model = train_gekc(config, args)