

import os 
import pandas as pd 
import torch 
import torch_geometric as pyg 
import numpy as np 
import copy 

class GraphBuilder: 
    'tkg data graph builder'

    def __init__(self, root, relnames, val_idxs, test_idxs, verbose=True, target_key=('drug', 'targets', 'gene')): 
        '''
        
        Parameters
        ----------
        root : str
            The root directory where the csv files are stored.
        relnames : list
            A list of strings, each string is the name of a csv file.
        '''

        self.root = root
        self.relnames = relnames
        self.val_idxs = val_idxs
        self.test_idxs = test_idxs
        self.verbose = verbose
        self.target_key = target_key
        
        self.relation_dfs = [] 
        for relname in relnames: 
            self.relation_dfs.append(pd.read_csv(f'{self.root}/{relname}'))
        
        self.relations = pd.concat(self.relation_dfs, axis=0)

        # need to index the entities given all node names within each node type 
        nodetypes = np.unique(self.relations.src_type.tolist() + self.relations.dst_type.tolist())
        if self.verbose: print(f'Node types: {nodetypes}')

        # drop na 
        self.relations = self.relations.dropna(subset=['src', 'dst', 'src_type', 'dst_type', 'relation'])

        self.node2idx = {}
        self.node_names_dict = {}
        for i,ntype in enumerate(nodetypes):

            src_nodes = self.relations[self.relations.src_type == ntype].src.tolist()
            dst_nodes = self.relations[self.relations.dst_type == ntype].dst.tolist()
            all_nodes = np.unique(src_nodes + dst_nodes)
            self.node2idx[ntype] = {node: i for i, node in enumerate(all_nodes)}
            self.node_names_dict[ntype] = all_nodes

        src_idx = [self.node2idx[ntype][node] for ntype, node in zip(self.relations.src_type, self.relations.src)]
        dst_idx = [self.node2idx[ntype][node] for ntype, node in zip(self.relations.dst_type, self.relations.dst)]

        self.relations = self.relations.assign(src_idx = src_idx)
        self.relations = self.relations.assign(dst_idx = dst_idx)


    def build(self):

        data = pyg.data.HeteroData()

        reltypes = self.relations[['src_type', 'dst_type', 'relation']].drop_duplicates().dropna()
        
        edge_index_dict = {} 
        reltype_keys = []
        for i,row in reltypes.iterrows():
            rels = self.relations[lambda x: (x.src_type == row.src_type) & (x.dst_type == row.dst_type) & (x.relation == row.relation)]

            key = (row.src_type, row.relation, row.dst_type)
            reltype_keys.append(key)

            edge_index = torch.stack([torch.tensor(rels.src_idx.tolist()), 
                                      torch.tensor(rels.dst_idx.tolist())], dim=0)
            
            edge_index_dict[key] = edge_index

        data.edge_index_dict = edge_index_dict
        data.node_name_dict = self.node_names_dict
        data.num_nodes_dict = {k: len(v) for k,v in self.node_names_dict.items()}
        data.edge_reltype = {k: np.array([i]*edge_index_dict[k].size(1)) for i,k in enumerate(reltype_keys)}

        assert len(data.edge_reltype) == len(data.edge_index_dict), 'Number of relations does not match number of edge index keys, are there nonunique relation names for different nodes?'

        self.data = data

    def get_triples(self): 
        
        data = copy.deepcopy(self.data)
        rel2int = {k:v[0] for k,v in self.data.edge_reltype.items()}

        train = {'head_type':[], 'head':[], 'relation':[], 'tail_type':[], 'tail':[]}
        valid = {'head_type':[], 'head':[], 'relation':[], 'tail_type':[], 'tail':[]}
        test = {'head_type':[], 'head':[], 'relation':[], 'tail_type':[], 'tail':[]}

        val_idxs = self.val_idxs 
        test_idxs = self.test_idxs
        n_target_edges = data.edge_index_dict[self.target_key].size(1)
        train_mask = np.ones((n_target_edges,), dtype=bool)
        train_mask[val_idxs] = False
        train_mask[test_idxs] = False
        train_idxs = train_mask.nonzero()[0]
        n_train = len(train_idxs)
        n_valid = len(val_idxs)
        n_test = len(test_idxs)

        # add train target edges 
        head_type, rel_type, tail_type = self.target_key
        target_relint = rel2int[self.target_key]
        train['head_type'] = [head_type]*n_train
        train['tail_type'] = [tail_type]*n_train
        train['head'] = data.edge_index_dict[self.target_key][0, train_idxs].detach().cpu().numpy().astype(int).tolist()
        train['tail'] = data.edge_index_dict[self.target_key][1, train_idxs].detach().cpu().numpy().astype(int).tolist()
        train['relation'] = [target_relint]*n_train

        # add all other training edges (all other relations)
        for key in data['edge_index_dict'].keys(): 
            if (key == self.target_key): 
                continue
            _head_type, _rel_type, _tail_type = key 
            _relint = rel2int[key]
            
            head, tail = data['edge_index_dict'][key].detach().cpu().numpy()
            train['head_type'] += [_head_type]*len(head)
            train['tail_type'] += [_tail_type]*len(tail)
            train['head'] += head.astype(int).tolist()
            train['tail'] += tail.astype(int).tolist()
            train['relation'] += [_relint]*len(head)

        # cast train types to numpy 
        train['head_type'] = np.array(train['head_type'])
        train['tail_type'] = np.array(train['tail_type'])
        train['head'] = np.array(train['head'])
        train['tail'] = np.array(train['tail'])
        train['relation'] = np.array(train['relation'])

        # add target edges to test/val
        edge_index = data.edge_index_dict[self.target_key]

        valid['head_type'] = np.array([head_type]*n_valid)
        valid['tail_type'] = np.array([tail_type]*n_valid)
        valid['head'] = np.array(edge_index[0, val_idxs].detach().cpu().numpy().astype(int).tolist())
        valid['tail'] = np.array(edge_index[1, val_idxs].detach().cpu().numpy().astype(int).tolist())
        valid['relation'] = np.array([target_relint]*n_valid).astype(int)

        test['head_type'] = np.array([head_type]*n_test)
        test['tail_type'] = np.array([tail_type]*n_test)
        test['head'] = np.array(edge_index[0, test_idxs].detach().cpu().numpy().astype(int).tolist())
        test['tail'] = np.array(edge_index[1, test_idxs].detach().cpu().numpy().astype(int).tolist())
        test['relation'] = np.array([target_relint]*n_test).astype(int)

        # Remove val/test edges from edge_index_dict 
        edge_index = data.edge_index_dict[self.target_key]
        edge_index = edge_index[:, train_idxs]
        data.edge_index_dict[self.target_key] = edge_index

        #data['edge_index_dict'] = data.edge_index_dict
        #del data.edge_index_dict

        #num_nodes_dict = data.node_nums_dict
        #del data.num_nodes_dict
        #for k,v in num_nodes_dict.items(): 
        #    data['num_nodes_dict'][k] = v

        # validate
        for triples in [train, valid, test]: 
            for key1 in triples:
                for key2 in triples: 
                    assert len(triples[key1]) == len(triples[key2]), f'length of {key1} is not same as length of {key2}'

        return train, valid, test, data



            






