'''
obg-biokg
'''



from torch_geometric.data import download_url, extract_tar, extract_gz
import argparse
import shutil
import os
import torch_geometric as pyg
import numpy as np
import torch
from sklearn.model_selection import KFold
import copy
import sys 
from ogb.linkproppred import LinkPropPredDataset


def get_args():
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", type=str, default='../data/biokg/', 
                        help="path to the save root dir")

    args = parser.parse_args()
    return args 
        

def add_edges(data, triples, rel2type): 
    # add edges from triples to data
    for rel in np.unique(triples['relation']): 
        rel_idxs = (triples['relation'] == rel).nonzero()[0]
        edge_type = rel2type[rel]
        data['edge_index_dict'][edge_type] = np.concatenate((data['edge_index_dict'][edge_type],
                                                        np.stack((triples['head'][rel_idxs], triples['tail'][rel_idxs]), axis=0)), axis=-1)
    return data 

if __name__ == '__main__': 

    args = get_args()
    print(args)

    os.makedirs(args.root + '/processed/', exist_ok=True)

    dataset = LinkPropPredDataset(name = 'ogbl-biokg', root=args.root + '/raw/')

    split_edge = dataset.get_edge_split()
    train_triples, valid_triples, test_triples = split_edge["train"], split_edge["valid"], split_edge["test"]
    data = dataset[0]

    data_ = pyg.data.HeteroData()
    data_.edge_index_dict = data['edge_index_dict']
    data_.edge_reltype = data['edge_reltype']
    data_.num_nodes_dict = data['num_nodes_dict']

    node_name_dict = {} 
    for k, v in data_.num_nodes_dict.items():
        node_name_dict[k] = np.array([f'{k}_{i}' for i in range(v)])
    data_.node_name_dict = node_name_dict 

    # sample negative edges <- deprecated... 

    rel2type = {v.item(0):k for k,v in data['edge_reltype'].items()}

    # data object contains train edges only; need to add valid and test edges to avoid sampling positive edges for evaluation
    # data = add_edges(data, valid_triples, rel2type)
    # data = add_edges(data, test_triples, rel2type)

    neg_valid = {'head': None,
                'tail': None, 
                'relation' : None}
        
    neg_test = {'head':None,
                    'tail':None, 
                    'relation' : None}

    torch.save(data_, args.root + '/processed/Data.pt')
    torch.save(train_triples, args.root + '/processed/pos_train.pt')
    torch.save(valid_triples, args.root + '/processed/pos_valid.pt')
    torch.save(test_triples, args.root + '/processed/pos_test.pt')
    torch.save(neg_valid, f'{args.root}/processed/neg_valid.pt')
    torch.save(neg_test, f'{args.root}/processed/neg_test.pt')