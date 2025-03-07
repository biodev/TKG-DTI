import argparse 
import numpy as np 
from tkgdti.train.train_gekc import train_gekc
import pandas as pd 
import os 
import copy 

# NOTE: this only works for the TKG dataset

def get_args(): 
     
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default='../data/tkg/processed/FOLD_0/', 
                        help="path to the save root dir")
    parser.add_argument("--out", type=str, default='../output/hparam_run/',
                        help="path to the save root dir")
    parser.add_argument("--n_epochs", type=int, default=100, 
                        help="number of epochs")
    parser.add_argument("--num_workers", type=int, default=10,
                        help="number of workers")
    parser.add_argument("--patience", type=int, default=10,
                        help="early stopping patience")
    
    args = parser.parse_args()
    args.verbose = False 
    args.remove_relation_idx = None 
    args.heteroA = False 
    args.log_every = 1
    args.use_cpu = False
    args.target_relation = tuple('drug,targets,gene'.split(','))
    args.compile = False
    args.target_metric = 'mrr'
    return args 


def get_config(): 
    # randomly sample from search space 

    _lr             = [1e-2, 1e-3, 1e-4]
    _wd             = [0, 1e-8, 1e-7, 1e-6]
    _channels       = [1024]
    _batch_size     = [5000, 10000, 25000]
    _dropout        = [0.]

    config = {'lr'                      : float( np.random.choice(_lr) ),
                'wd'                    : float( np.random.choice(_wd) ),
                'channels'              : int( np.random.choice(_channels) ),
                'batch_size'            : int( np.random.choice(_batch_size) ),
                'dropout'               : float( np.random.choice(_dropout) ),
                'checkpoint'            : False, 
                'bias'                  : False,
                'optim'                 : 'adam',
                'lr_scheduler'          : None,
                'remove_relation_idx'   : None}
    
    return config
        

if __name__ == '__main__': 

        args = get_args()
        config = get_config()

        # update args with config 
        args.batch_size = config['batch_size']

        os.makedirs(args.out, exist_ok=True)
        
        print()
        print('#'*50)
        print('CONFIG: ')
        print() 
        print(config)
        print() 
        print(args)
        print()
        print('#'*50)
        print()

        config['num_workers']   = args.num_workers
        config['n_epochs']      = args.n_epochs

        ## 
        
        uid, val_metrics, test_metrics = train_gekc(config, copy.deepcopy(args))

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