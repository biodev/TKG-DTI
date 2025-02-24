

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
                 filter_to_relation=None):
        """

        """
        self.channels = channels
        self.num_node_dict = num_node_dict
        self.edge_index_dict = edge_index_dict
        self.target_relation = target_relation

        self.pos_heads = torch.tensor(triples['head'], dtype=torch.long)
        self.pos_tails = torch.tensor(triples['tail'], dtype=torch.long)
        self.pos_relations = torch.tensor(triples['relation'], dtype=torch.long)

        if filter_to_relation is not None: 
            idxs = torch.isin(self.pos_relations, torch.tensor(filter_to_relation, dtype=torch.long)).nonzero(as_tuple=True)[0]
            self.pos_heads = self.pos_heads[idxs]
            self.pos_tails = self.pos_tails[idxs]
            self.pos_relations = self.pos_relations[idxs]

        self.x_dict = {node:torch.zeros((num_nodes, self.channels), dtype=torch.float32) for node, num_nodes in self.num_node_dict.items()}

    def __len__(self):

        return len(self.pos_heads)

    def __getitem__(self, idx):

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