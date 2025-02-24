

import torch 
from torch_geometric.data import Dataset
import sys 
import numpy as np
sys.path.append('../')
import copy
import torch_geometric as pyg

class TriplesDatasetGNN(Dataset):
    """"""

    def __init__(self, triples, edge_index_dict, channels, num_node_dict, target_relation, 
                 filter_to_relation=None, training=False, prop_train_predict = 0.25):
        """

        """
        self.channels = channels
        self.num_node_dict = num_node_dict
        self.edge_index_dict = edge_index_dict
        self.target_relation = target_relation

        self.pos_heads = torch.tensor(triples['head'], dtype=torch.long)
        self.pos_tails = torch.tensor(triples['tail'], dtype=torch.long)
        self.pos_relations = torch.tensor(triples['relation'], dtype=torch.long)

        self.training = training 
        self.PROP_TRAIN_PREDICT = prop_train_predict

        if filter_to_relation is not None: 
            idxs = torch.isin(self.pos_relations, torch.tensor(filter_to_relation, dtype=torch.long)).nonzero(as_tuple=True)[0]
            self.pos_heads = self.pos_heads[idxs]
            self.pos_tails = self.pos_tails[idxs]
            self.pos_relations = self.pos_relations[idxs]

        self.x_dict = {node:torch.zeros((num_nodes, self.channels), dtype=torch.float32) for node, num_nodes in self.num_node_dict.items()}

        if self.training: 
            self.n_drugs = torch.unique(self.pos_heads).shape[0]
            self.drug2idxs = {i: (self.pos_heads == i).nonzero(as_tuple=True)[0] for i in range(self.n_drugs)} 

    def __len__(self):

        if self.training: 
            return self.n_drugs
        else:
            return len(self.pos_heads)
            
        
    def _eval_getitem(self, idx):

        pos_head = self.pos_heads[idx].detach()
        pos_tail = self.pos_tails[idx].detach()
        pos_relation = self.pos_relations[idx].detach()

        x_dict = copy.deepcopy(self.x_dict)
        x_dict[self.target_relation[0]][pos_head] = torch.ones((self.channels,), dtype=torch.float32)

        edge_index_dict = copy.deepcopy(self.edge_index_dict)
        edge_index = edge_index_dict[self.target_relation]
        mask = ~((edge_index[0] == pos_head) & (edge_index[1] == pos_tail))   # remove training data edge from edge_index
        edge_index_dict[self.target_relation] = edge_index[:, mask]

        y_dict = {node:torch.zeros((1,num_nodes), dtype=torch.float32) for node, num_nodes in self.num_node_dict.items()}

        y = torch.zeros((1, self.num_node_dict[self.target_relation[-1]]), dtype=torch.float32)
        y[:, pos_tail] = 1
        y_dict[self.target_relation[-1]] = y   

        # Create a HeteroData object for this instance
        data = pyg.data.HeteroData()
        
        # add y features 
        for node_type in y_dict:
            data[node_type].y = y_dict[node_type].view(-1)

        # Add node features
        for node_type in x_dict:
            data[node_type].x = x_dict[node_type]

        # Add edge indices
        for edge_type in edge_index_dict:
            src_type, relation, dst_type = edge_type
            data[(src_type, relation, dst_type)].edge_index = edge_index_dict[edge_type]

        data.y_protein = y
        data.head = torch.tensor([pos_head], dtype=torch.long)
        data.tail = torch.tensor([pos_tail], dtype=torch.long)
        data.relation = torch.tensor([pos_relation], dtype=torch.long)

        return data
    
    def _train_getitem(self, idx):
        # idx refers to drug index (int)

        triple_idxs = self.drug2idxs[idx]

        # select 0.5 of the triples for this drug
        triple_idxs = triple_idxs[torch.randperm(triple_idxs.shape[0])[:int(triple_idxs.shape[0]*self.PROP_TRAIN_PREDICT)]]

        pos_heads = self.pos_heads[triple_idxs].detach()
        pos_tails = self.pos_tails[triple_idxs].detach()
        pos_relations = self.pos_relations[triple_idxs].detach()

        x_dict = copy.deepcopy(self.x_dict)

        edge_index_dict = copy.deepcopy(self.edge_index_dict)
        for pos_head, pos_tail in zip(pos_heads, pos_tails):
            edge_index = edge_index_dict[self.target_relation]
            x_dict[self.target_relation[0]][pos_head] = torch.ones((self.channels,), dtype=torch.float32)
            mask = ~((edge_index[0] == pos_head) & (edge_index[1] == pos_tail))   # remove training data edge from edge_index
            edge_index_dict[self.target_relation] = edge_index[:, mask]

        y_dict = {node:torch.zeros((1,num_nodes), dtype=torch.float32) for node, num_nodes in self.num_node_dict.items()}

        y = torch.zeros((1, self.num_node_dict[self.target_relation[-1]]), dtype=torch.float32)
        y[:, pos_tails] = 1
        y_dict[self.target_relation[-1]] = y   

        # Create a HeteroData object for this instance
        data = pyg.data.HeteroData()
        
        # add y features 
        for node_type in y_dict:
            data[node_type].y = y_dict[node_type].view(-1)

        # Add node features
        for node_type in x_dict:
            data[node_type].x = x_dict[node_type]

        # Add edge indices
        for edge_type in edge_index_dict:
            src_type, relation, dst_type = edge_type
            data[(src_type, relation, dst_type)].edge_index = edge_index_dict[edge_type]

        data.y_protein = y
        data.head = pos_heads #torch.tensor([pos_heads], dtype=torch.long)
        data.tail = pos_tails #torch.tensor([pos_tails], dtype=torch.long)
        data.relation = pos_relations #torch.tensor([pos_relations], dtype=torch.long)

        return data



    def __getitem__(self, idx):

        if self.training: 
            data = self._train_getitem(idx)

        else: 
            data = self._eval_getitem(idx)

        return data