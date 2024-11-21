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
from tkgdti.data.negative_sampling import negative_sampling


def get_args():
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", type=str, default='../data/biokg/', 
                        help="path to the save root dir")
    parser.add_argument("--n_neg_samples", type=int, default=500, help="number of negative samples")

    args = parser.parse_args()
    return args 


def sample_neg_edges(triples, data, rel2type, nsamples): 
    nedges = len(triples['head'])
    neg_heads = torch.zeros((nedges, int(nsamples*2)), dtype=torch.long)
    neg_tails = torch.zeros((nedges, int(nsamples*2)), dtype=torch.long)
    neg_relations = torch.zeros((nedges, int(nsamples*2)), dtype=torch.long)

    for rel in np.unique(triples['relation']): 
        print('making negative samples, relation:', rel, end='\r')
        rel_idxs = (triples['relation'] == rel).nonzero()[0]
        edge_type = rel2type[rel]
        head_type, _ , tail_type = edge_type
        neg_heads[rel_idxs], neg_tails[rel_idxs], neg_relations[rel_idxs] = negative_sampling(heads = triples['head'][rel_idxs],
                                                                                                tails = triples['tail'][rel_idxs],
                                                                                                edge_index   = data['edge_index_dict'][edge_type], 
                                                                                                size         = (data['num_nodes_dict'][head_type], data['num_nodes_dict'][tail_type]), 
                                                                                                relint       = rel, 
                                                                                                n            = nsamples)
        
    return neg_heads.detach(), neg_tails.detach(), neg_relations.detach()

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

    # sample negative edges 
    # negative sampling ... 
    rel2type = {v.item(0):k for k,v in data['edge_reltype'].items()}

    # data object contains train edges only; need to add valid and test edges to avoid sampling positive edges for evaluation
    data = add_edges(data, valid_triples, rel2type)
    data = add_edges(data, test_triples, rel2type)

    valid_neg_heads, valid_neg_tails, valid_neg_relations = sample_neg_edges(valid_triples, data, rel2type, nsamples=args.n_neg_samples)
    test_neg_heads, test_neg_tails, test_neg_relations = sample_neg_edges(test_triples, data, rel2type, nsamples=args.n_neg_samples)

    neg_valid = {'head': valid_neg_heads,
                     'tail': valid_neg_tails, 
                     'relation' : valid_neg_relations}
        
    neg_test = {'head': test_neg_heads,
                    'tail': test_neg_tails, 
                    'relation' : test_neg_relations}

    torch.save(data, args.root + '/processed/Data.pt')
    torch.save(train_triples, args.root + '/processed/pos_train.pt')
    torch.save(valid_triples, args.root + '/processed/pos_valid.pt')
    torch.save(test_triples, args.root + '/processed/pos_test.pt')
    torch.save(neg_valid, f'{args.root}/processed/neg_valid.pt')
    torch.save(neg_test, f'{args.root}/processed/neg_test.pt')