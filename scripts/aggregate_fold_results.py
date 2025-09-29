

import pandas as pd 
import os 
import argparse 

def get_args(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='.')
    return parser.parse_args()

def main(args): 
    
    test_results = []; val_results = []
    for dir in os.listdir(args.root): 
        if os.path.isdir(os.path.join(args.root, dir)): 
            
            test_res_path = os.path.join(args.root, dir, 'test_metrics.csv')
            val_res_path = os.path.join(args.root, dir, 'valid_metrics.csv') 

            if os.path.exists(test_res_path):  
                test_results.append(pd.read_csv(test_res_path))
            else: 
                print(f'{test_res_path} does not exist')
            if os.path.exists(val_res_path): 
                val_results.append(pd.read_csv(val_res_path))
            else: 
                print(f'{val_res_path} does not exist')

    test_results = pd.concat(test_results)
    val_results = pd.concat(val_results)

    test_results.to_csv(os.path.join(args.root, 'test_results_aggregated.csv'), index=False)
    val_results.to_csv(os.path.join(args.root, 'val_results_aggregated.csv'), index=False) 


if __name__ == '__main__': 
    args = get_args()
    main(args) 

