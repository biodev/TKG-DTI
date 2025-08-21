'''
DTINet knowledge graph (HeteroA): https://github.com/luoyunan/PyDTINet/blob/main/data.tar.gz


`ParentData` will be a torch geometric hetero data object with ALL edges
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
#from tkgdti.data.negative_sampling import negative_sampling

TARGET_KEY = ('drug', 'drug->target->protein', 'protein')

def get_args():
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", type=str, default='../data/HeteroA/', 
                        help="path to the save root dir")
    parser.add_argument("--url", type=str, default='https://github.com/luoyunan/PyDTINet/raw/main/data.tar.gz', 
                        help="url to DTINet KG data")
    parser.add_argument("--k", type=int, default=10, 
                        help="number of folds to create")
    parser.add_argument("--train_p", type=float, default=0.9, help="train partition proportion (rest will be placed in validation set)")
    #parser.add_argument("--use_sim", action='store_true', default=False, help='whether to binarize the similarity matrices.')
    parser.add_argument("--pp_thresh", type=float, default=90, help="threshold to binarize the protein-protein sequence similarity network")
    parser.add_argument("--dd_thresh", type=float, default=0.9, help="threshold to binarize the drug-drug similarity network")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--n_neg_samples", type=int, default=500, help="number of negative samples")

    args = parser.parse_args()
    return args 

def load_mat(mat_path): 

    entityA_path, entityB_path = mat_path[:-4].split('_')[-2:]
    root = '/'.join(mat_path.split('/')[:-1])
    entityA = [x.strip() for x in open(root + '/' + entityA_path + '.txt', 'r').readlines()]
    entityB = [x.strip() for x in open(root + '/' + entityB_path + '.txt', 'r').readlines()]
    A = np.loadtxt(mat_path)
    edge_index = np.stack(A.nonzero(), axis=0)

    # BUG?: The upper and lower triangle of the adjacency matrices don't match; not necessarily an issue but little strange. 
    # check: are edges unique? I assume so, bc selecting upper triangle doesn't return the right number of edges according to DTINet paper
    # print('sum triu', np.triu(A, k=1).sum())
    # print('sum tril', np.tril(A, k=1).sum())

    return edge_index, entityA, entityB , entityA_path, entityB_path

def make_data(args): 

    data = pyg.data.HeteroData()

    data.ordered_relations = []
    edge_reltype = {}
    node_name_dict = {}
    num_nodes_dict = {}
    for i, (mat_path,edgetype) in enumerate(zip(['mat_drug_disease.txt',
                                                'mat_protein_protein.txt',
                                                #'mat_protein_drug.txt',   #NOTE: equivalent to transposed `mat_drug_protein.txt`
                                                'mat_protein_disease.txt',
                                                'mat_drug_drug.txt',
                                                'mat_drug_se.txt',
                                                'mat_drug_protein.txt',
                                                'mat_protein_protein.txt', # sequence similarity edges 
                                                'mat_drug_drug.txt'],      # sequence similarity edges 
                                                ['association',
                                                'interaction',
                                                #'targets',   #NOTE: equivalent to transposed `mat_drug_protein.txt`
                                                'association',
                                                'interaction',
                                                'association',
                                                'target',
                                                'similarity', 
                                                'similarity'])): 
        
        edge_index, entityA, entityB, entityA_name, entityB_name = load_mat(args.root + '/raw/' +  mat_path)

        data.ordered_relations.append((entityA_name, '->'.join([entityA_name,edgetype,entityB_name]), entityB_name))
        
        data['edge_index_dict'][(entityA_name, '->'.join([entityA_name,edgetype,entityB_name]), entityB_name)] = edge_index 
        edge_reltype[(entityA_name, '->'.join([entityA_name,edgetype,entityB_name]), entityB_name)] = i*np.ones(edge_index.shape[1], dtype=int) 
        node_name_dict[entityA_name] = entityA
        node_name_dict[entityB_name] = entityB
        num_nodes_dict[entityA_name] = len(entityA)
        num_nodes_dict[entityB_name] = len(entityB)

        print(f'{",".join(data.ordered_relations[-1])} {(50-len("".join(data.ordered_relations[-1])))*"-"}> num edges: {edge_index.shape[1]} \t\t| num_nodes (A): {len(entityA)} \t\t| num_nodes (B): {len(entityB)}')

    data.edge_reltype = edge_reltype
    data.num_nodes_dict = num_nodes_dict
    data.node_name_dict = node_name_dict

    return data

def proc_sim2adj(args): 
    '''
    Binarizing similarity matrices to create adj matrix compatible with our learning algorithms 
    '''
    pp_sim_path = args.root + '/raw/Similarity_Matrix_Proteins.txt'
    dd_sim_path = args.root + '/raw/Similarity_Matrix_Drugs.txt'
    pp_mat_path = args.root + '/raw/mat_protein_protein.txt'
    dd_mat_path = args.root + '/raw/mat_drug_drug.txt'

    pp_sim_mat = np.loadtxt(pp_sim_path)
    dd_sim_mat = np.loadtxt(dd_sim_path)

    ppA = 1*(pp_sim_mat > args.pp_thresh)
    ddA = 1*(dd_sim_mat > args.dd_thresh)

    np.savetxt(pp_mat_path, ppA.astype(int))
    np.savetxt(dd_mat_path, ddA.astype(int))


if __name__ == '__main__': 

    args = get_args()
    print(args)

    if os.path.exists(args.root): shutil.rmtree(args.root)
    os.makedirs(args.root, exist_ok=True)

    download_url(args.url, args.root + '/raw/')
    extract_tar(args.root + 'raw/data.tar.gz', args.root)
    shutil.copytree(args.root + '/data/', args.root + '/raw/', dirs_exist_ok=True)
    shutil.rmtree(args.root + '/data/')
    os.remove(args.root + 'raw/data.tar.gz') 

    # QUESTION: DTINet uses the similarity matrices directly; HGAN does not appear to use the similarity matrices. 
    proc_sim2adj(args)

    # NOTE: HeteroA uses bidirectional edges for drug->target and target->drug 
    data = make_data(args)
   
    os.makedirs(args.root + '/processed/', exist_ok=True)

    torch.save(data, f'{args.root}/processed/ParentData.pt')

    # NOTE: only args.target_relint (DTI relation = 5) will be placed in test/val
    # NOTE: test set will be assigned with K-fold CV procedure, however, validation set will be randomly sampled from train. 

    rel2type = {v.item(0):k for k,v in data['edge_reltype'].items()}
    type2rel = {v:k for k,v in rel2type.items()}
    target_relint = type2rel[TARGET_KEY]
    head_type, rel_type, tail_type = TARGET_KEY  
    print(f'Using target relation index: {target_relint} ({TARGET_KEY})')

    edge_index = data['edge_index_dict'][TARGET_KEY]
    nedges = edge_index.shape[0]

    kf = KFold(n_splits=args.k, shuffle=True, random_state=args.seed)

    print()
    for i, (train_index, test_index) in enumerate(kf.split(edge_index.T)):

        print('generating FOLD: ', i, end='\r')

        os.makedirs(f'{args.root}/processed/FOLD_{i}', exist_ok=True)

        # randomly sample validation set from train set 
        n_train = int(args.train_p * len(train_index))
        n_valid = len(train_index) - n_train 
        n_test = len(test_index)

        np.random.shuffle(train_index) # shuffle in place
        valid_index = train_index[:n_valid]
        train_index = train_index[n_valid:]

        # remove val/test edges from data 
        edge_index_fold = copy.deepcopy(edge_index)[:, train_index]
        
        # remove all valid/test edges from this fold `data` object (KG graph)
        data_fold = copy.deepcopy(data)
        data_fold['edge_index_dict'][TARGET_KEY] = edge_index_fold

        train = {'head_type':[], 'head':[], 'relation':[], 'tail_type':[], 'tail':[]}
        valid = {'head_type':[], 'head':[], 'relation':[], 'tail_type':[], 'tail':[]}
        test = {'head_type':[], 'head':[], 'relation':[], 'tail_type':[], 'tail':[]}

        # add train target edges 
        train['head_type'] = [head_type]*n_train
        train['tail_type'] = [tail_type]*n_train
        train['head'] = edge_index[0, train_index].astype(int).tolist()
        train['tail'] = edge_index[1, train_index].astype(int).tolist()
        train['relation'] = [target_relint]*n_train

        # add all other training edges (all other relations)
        for key in data['edge_index_dict'].keys(): 
            if (key == TARGET_KEY): 
                continue
            _head_type, _, _tail_type = key 
            _relint = type2rel[key]
            
            head, tail = data['edge_index_dict'][key]
            train['head_type'] += [_head_type]*len(head)
            train['tail_type'] += [_tail_type]*len(tail)
            train['head'] += head.tolist()
            train['tail'] += tail.astype(int).tolist()
            train['relation'] += [_relint]*len(head)

        # cast train types to numpy 
        train['head_type'] = np.array(train['head_type'])
        train['tail_type'] = np.array(train['tail_type'])
        train['head'] = np.array(train['head'])
        train['tail'] = np.array(train['tail'])
        train['relation'] = np.array(train['relation'])

        # add target edges to test/val
        valid['head_type'] = np.array([head_type]*n_valid)
        valid['tail_type'] = np.array([tail_type]*n_valid)
        valid['head'] = np.array(edge_index[0, valid_index].astype(int).tolist())
        valid['tail'] = np.array(edge_index[1, valid_index].astype(int).tolist())
        valid['relation'] = np.array([target_relint]*n_valid).astype(int)

        test['head_type'] = np.array([head_type]*n_test)
        test['tail_type'] = np.array([tail_type]*n_test)
        test['head'] = np.array(edge_index[0, test_index].astype(int).tolist())
        test['tail'] = np.array(edge_index[1, test_index].astype(int).tolist())
        test['relation'] = np.array([target_relint]*n_test).astype(int)

        # validate
        for triples in [train, valid, test]: 
            for key1 in triples:
                for key2 in triples: 
                    assert len(triples[key1]) == len(triples[key2]), f'length of {key1} is not same as length of {key2}'

        # save data/idxs 
        np.save(f'{args.root}/processed/FOLD_{i}/train_idxs.npy', train_index)
        np.save(f'{args.root}/processed/FOLD_{i}/valid_idxs.npy', valid_index)
        np.save(f'{args.root}/processed/FOLD_{i}/test_idxs.npy', test_index)

        # 
        torch.save(train, f'{args.root}/processed/FOLD_{i}/pos_train.pt') 
        torch.save(valid, f'{args.root}/processed/FOLD_{i}/pos_valid.pt') 
        torch.save(test, f'{args.root}/processed/FOLD_{i}/pos_test.pt') 
        
        torch.save(data_fold, f'{args.root}/processed/FOLD_{i}/Data.pt')
        
        neg_valid = {'head': None,
                     'tail': None, 
                     'relation' : None}
        
        neg_test = {'head':None,
                     'tail':None, 
                     'relation' : None}

        torch.save(neg_valid, f'{args.root}/processed/FOLD_{i}/neg_valid.pt')
        torch.save(neg_test, f'{args.root}/processed/FOLD_{i}/neg_test.pt')

