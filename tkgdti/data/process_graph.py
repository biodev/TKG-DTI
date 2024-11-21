
import torch
import torch_geometric as pyg
import copy
from torch_geometric.data import HeteroData, Batch 

def process_graph(data): 

    tdata = pyg.data.HeteroData()

    for key, edge_index  in data['edge_index_dict'].items(): 
        
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        if key[0] == key[2]: 
            if not pyg.utils.is_undirected(edge_index): 
                edge_index = pyg.utils.to_undirected(edge_index)

        tdata[key].edge_index = edge_index 

    for node, names in data['node_name_dict'].items(): 
        tdata[node].names = names 
        tdata[node].num_nodes = len(names)

    row,col = torch.tensor(data['edge_index_dict']['protein', 'protein->association->disease', 'disease'], dtype=torch.long)
    rev_edge_index = torch.stack([col, row], dim=0)
    tdata['disease', 'disease->association->protein', 'protein'].edge_index = rev_edge_index

    row,col = torch.tensor(data['edge_index_dict']['drug', 'drug->association->se', 'se'], dtype=torch.long)
    rev_edge_index = torch.stack([col, row], dim=0)
    tdata['se', 'se->association->drug', 'drug'].edge_index = rev_edge_index 

    del tdata['edge_index']
    del tdata['num_nodes']
    del tdata['names']

    return tdata 

'''
def batch_data(heads, tails, channels, edge_index_dict):
    
    data_list = []
    for h, t in zip(heads, tails):
        # Create copies of x_dict and edge_index_dict for each instance
        x_dict = copy.deepcopy(self.x_dict_)
        x_dict = {k: v.to(h.device) for k, v in x_dict.items()}
        x_dict['drug'][h] = torch.ones((self.hidden_channels,), dtype=torch.float32)

        # Modify edge_index_dict to remove the edge between h and t
        edge_index_dict = copy.deepcopy(self.edge_index_dict)
        edge_index = edge_index_dict['drug', 'drug->target->protein', 'protein']
        mask = ~((edge_index[0] == h) & (edge_index[1] == t))
        edge_index_dict['drug', 'drug->target->protein', 'protein'] = edge_index[:, mask]

        if self.training and self.edge_dropout > 0:
            edge_index_dict = {key: pyg.utils.dropout_edge(edge_index, p=self.edge_dropout, training=True)[0] for key, edge_index in edge_index_dict.items()}

        if requires_grad: 
            x_dict = {key: x.requires_grad_(True) for key, x in x_dict.items()} 

        # Create a HeteroData object for this instance
        data = HeteroData()

        # Add node features
        for node_type in x_dict:
            data[node_type].x = x_dict[node_type]

        # Add edge indices
        for edge_type in edge_index_dict:
            src_type, relation, dst_type = edge_type
            data[(src_type, relation, dst_type)].edge_index = edge_index_dict[edge_type]

        data_list.append(data)

    # Batch the list of HeteroData objects
    batch = Batch.from_data_list(data_list)

    return batch
'''