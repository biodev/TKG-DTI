

import argparse 
import numpy as np 

from tkgdti.train.train_gnn import train_gnn
import pandas as pd 
import os 
import copy 


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, default='../data/tkg/processed/FOLD_0', help="path to the data dir")
    parser.add_argument("--out", type=str, default='../output/GNN_HPARAM_TUNING/', help="path to output dir")
    parser.add_argument("--searchspace", type=str, default='hp', help="search space for tuning ['hp', 'kg']")
    parser.add_argument("--n_runs", type=int, default=100, help="number of runs to perform")

    # other params (same as in train_gnn)
    parser.add_argument("--n_epochs", type=int, default=100, help="number of training epochs")
    parser.add_argument("--num_workers", type=int, default=10, help="number of data loader workers")
    parser.add_argument("--verbose", action='store_true', default=False, help="verbosity")
    parser.add_argument("--checkpoint", action='store_true', default=False, help="whether to use gradient checkpointing")
    parser.add_argument("--log_every", default=1, type=int, help="log every n epochs")
    parser.add_argument("--compile", action='store_true', default=False, help="whether to compile the model")
    parser.add_argument("--patience", default=10, type=int, help="early stopping patience")
    parser.add_argument("--target_metric", default='mrr', type=str, help="metric to use for early stopping [hits@10, mrr, auroc]")
    parser.add_argument("--target_relation", default='drug,targets,gene', type=str, help="tuple key of target relation")
    parser.add_argument("--heteroA", action='store_true', default=False, help="whether the data is from the HeteroA dataset")

    args = parser.parse_args()
    args.target_relation = tuple(args.target_relation.split(','))

    return args

def get_config(searchspace): 

    if searchspace == 'hp': 

        # 3 x 4 x 3x 2 x 2 x 3 x 3 x 3 x 3 
        _lr             = [1e-2, 5e-3, 1e-3]
        _wd             = [0, 1e-8, 1e-7, 1e-6]
        _channels       = [8, 12, 16]
        _layers         = [3, 4]
        _batch_size     = [5, 10]
        _heads          = [1, 2, 3] 
        _norm           = ['layer']
        _conv           = ['gat']                               # only gat seems to work with GNNExplainer ... 
        _nonlin         = ['elu', 'gelu', 'mish']
        _dropout        = [0.]
        _edge_dim       = [2, 4, 6] 
        _residual       = [True, False]   

        config = {'lr'                      : float( np.random.choice(_lr) ),
                    'wd'                    : float( np.random.choice(_wd) ),
                    'channels'              : int( np.random.choice(_channels) ),
                    'layers'                : int( np.random.choice(_layers) ),
                    'batch_size'            : int( np.random.choice(_batch_size) ),
                    'heads'                 : int( np.random.choice(_heads) ),
                    'norm'                  : np.random.choice(_norm),
                    'conv'                  : np.random.choice(_conv),
                    'nonlin'                : np.random.choice(_nonlin),
                    'dropout'               : float( np.random.choice(_dropout) ),
                    'edge_dim'              : int( np.random.choice(_edge_dim) ),
                    'checkpoint'            :False, 
                    'residual'              : bool( np.random.choice(_residual) ), 
                    'bias'                  :False,
                    'remove_relation_idx'   :None}
        
        return config
        
    elif searchspace == 'kg':

        __NUM_RELATIONS__ = 49 

        pass 

    else:
        raise ValueError(f'Unrecognized searchspace: {searchspace} [options: hp, kg]')

if __name__ == '__main__': 

    args = get_args()

 
    for i in range(args.n_runs): 

        ## 

        config = get_config(args.searchspace)
        
        print()
        print('#'*50)
        print('CONFIG: ')
        print(config)
        print('#'*50)
        print()

        config['num_workers']   = args.num_workers
        config['n_epochs']      = args.n_epochs

        ## 
        
        uid, val_metrics, test_metrics = train_gnn(config, copy.deepcopy(args))

        val_metrics = {f'val_{k}':v for k,v in val_metrics.items()}
        test_metrics = {f'test_{k}':v for k,v in test_metrics.items()}

        # save results to disk 
        res = pd.DataFrame({**config, **val_metrics, **test_metrics, 'uid':uid}, index=[0])

        if os.path.exists(args.out + '/hparam_test_results.csv'): 
            res.to_csv(args.out + '/hparam_test_results.csv', mode='a', header=False, index=False)
        else: 
            res.to_csv(args.out + '/hparam_test_results.csv', mode='w', header=True, index=False)

        # delete uid folder 
        os.system(f'rm -r {args.out}/{uid}')
        os.system(f'rm -r {args.out}/test_metrics.csv')
        os.system(f'rm -r {args.out}/valid_metrics.csv')