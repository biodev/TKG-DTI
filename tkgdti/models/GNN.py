from torch_geometric.nn import GATConv, GATv2Conv
import torch
import torch_geometric as pyg
from tkgdti.models.DiffConv import DiffConv

def _get_conv(conv, channels, heads, bias, edge_dim, dropout, nonlin, add_self_loops=True):

    if conv == 'gatv2':
        conv = lambda: pyg.nn.GATv2Conv(-1, channels, 
                                    heads=heads, 
                                    bias=bias, 
                                    concat=True,
                                    add_self_loops=True,
                                    edge_dim=edge_dim, 
                                    dropout=dropout)
    elif conv == 'gat':
        conv = lambda: pyg.nn.GATConv(-1, channels, 
                                    heads=heads, 
                                    bias=bias, 
                                    concat=True,
                                    add_self_loops=True, 
                                    edge_dim=edge_dim, 
                                    dropout=dropout)
    elif conv == 'transformer':
        conv = lambda: pyg.nn.TransformerConv(-1, channels, 
                                                heads=heads, 
                                                bias=bias, 
                                                edge_dim=edge_dim, 
                                                concat=True, 
                                                dropout=dropout, 
                                                root_weight=True)
    elif conv == 'gine': 
        conv = lambda: pyg.nn.GINEConv(nn=torch.nn.Sequential(torch.nn.Linear(channels, channels, bias=False), 
                                                                nonlin(), 
                                                                torch.nn.Dropout(dropout), 
                                                                torch.nn.Linear(channels, channels, bias=False)), 
                                    edge_dim=edge_dim)
    elif conv == 'nnconv':  
        assert channels < 20, 'NNConv is very memory intensive, fewer than 20 channels recommended'
        conv = lambda: pyg.nn.NNConv(channels, channels, nn=torch.nn.Sequential(torch.nn.Linear(edge_dim, channels), 
                                                                                nonlin(), 
                                                                                torch.nn.Dropout(dropout),
                                                                                torch.nn.Linear(channels, channels*channels)), 
                                                        aggr='mean')
        
    elif conv == 'diff':
        conv = lambda: DiffConv(in_channels=channels, out_channels=channels, edge_channels=edge_dim)

    else:
        raise ValueError(f'Unrecognized conv: {conv} [options: gat, gatv2, transformer, gine, nnconv]')
    
    return conv

def _get_norm(norm, channels):

    if norm == 'layer': 
        # NOTE: LayerNorm allows leak/dependencies between nodes independant of path, this may confound interpretation. 
        norm = lambda: pyg.nn.norm.LayerNorm(channels, mode='node', affine=False)
    elif norm == 'batch':
        #      BatchNorm would be more appropriate for interpretation, but we're limited to small batch sizes with mem constraints... 
        norm = lambda:  pyg.nn.norm.BatchNorm(channels, affine=False)
    elif norm == 'none':
        norm = lambda: torch.nn.Identity()
    else:
        raise ValueError(f'Unrecognized norm: {norm} [options: layer, batch, none]')
    
    return norm


class GNN(torch.nn.Module):
    def __init__(self, n_relations, channels, layers=3, dropout=0, heads=1, bias=True,
                        edge_dim=8, nonlin=torch.nn.ReLU,
                        checkpoint=True, residual=True, conv='gat', norm='none', norm_first=False, mlp_out_dim=100):
        super().__init__()

        self.checkpoint = checkpoint
        self.residual = residual 
        self.channels = channels
        self.norm_first = norm_first

        conv = _get_conv(conv, channels, heads, bias, edge_dim, dropout, nonlin)

        convs = [] 
        for i in range(layers): 
            convs.append(conv())
        self.convs = torch.nn.ModuleList(convs)    

        norm = _get_norm(norm, channels*heads)
        self.norms = torch.nn.ModuleList([norm() for i in range(layers)])
        
        self.edge_embedding = torch.nn.Embedding(n_relations, edge_dim)

        self.nonlin = nonlin() 

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(channels*heads, mlp_out_dim),
            nonlin(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(mlp_out_dim, 1)
        )
    
    def forward(self, x, edge_index, edge_type):
        
        # x is shape: (N, 1)
        x = x.expand(-1, self.channels)
        for i,conv in enumerate(self.convs):
            xl = x
            
            edge_attr = self.edge_embedding(edge_type)

            if self.checkpoint: 
                x = torch.utils.checkpoint.checkpoint(conv, x, edge_index, edge_attr, use_reentrant=False)
            else: 
                x = conv(x=x, edge_index=edge_index, edge_attr=edge_attr)

            if self.residual and (i > 0): 
                x = x + xl

            if self.norm_first: 
                x = self.norms[i](x)
                x = self.nonlin(x)
            else:
                x = self.nonlin(x)
                x = self.norms[i](x)

        return self.mlp(x).sigmoid()