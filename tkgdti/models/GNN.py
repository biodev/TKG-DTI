from torch_geometric.nn import GATConv, GATv2Conv
import torch
import torch_geometric as pyg



class GNN(torch.nn.Module):
    def __init__(self, n_relations, channels, layers=3, dropout=0, heads=1, bias=True,
                        edge_dim=8, norm_mode='graph', norm_affine=True, nonlin=torch.nn.ReLU,
                        checkpoint=True, residual=True, conv='gat'):
        super().__init__()

        self.checkpoint = checkpoint
        self.residual = residual 
        self.channels = channels

        if conv == 'gatv2':
            conv = lambda: pyg.nn.GATv2Conv(channels, channels, 
                                      heads=heads, 
                                      bias=bias, 
                                      concat=False,
                                      add_self_loops=False, 
                                      edge_dim=edge_dim, 
                                      dropout=dropout)
        elif conv == 'gat':
            conv = lambda: pyg.nn.GATConv(channels, channels, 
                                      heads=heads, 
                                      bias=bias, 
                                      concat=False,
                                      add_self_loops=False, 
                                      edge_dim=edge_dim, 
                                      dropout=dropout)
        elif conv == 'transformer':
            conv = lambda: pyg.nn.TransformerConv(-1, channels, heads=heads, bias=bias, edge_dim=edge_dim, dropout=dropout, root_weight=False)
        elif conv == 'gine': 
            conv = lambda: pyg.nn.GINEConv(nn=torch.nn.Sequential(torch.nn.Linear(channels, channels, bias=False), 
                                                                    nonlin(), 
                                                                    torch.nn.Dropout(dropout), 
                                                                    torch.nn.Linear(channels, channels, bias=False)), 
                                     edge_dim=edge_dim)
        elif conv == 'nnconv':  
            assert channels < 20, 'NNConv is very memory intensive for large channels'
            conv = lambda: pyg.nn.NNConv(channels, channels, nn=torch.nn.Sequential(torch.nn.Linear(edge_dim, channels), 
                                                                                    nonlin(), 
                                                                                    torch.nn.Dropout(dropout),
                                                                                    torch.nn.Linear(channels, channels*channels)), 
                                                            aggr='mean')
        else:
            raise ValueError(f'Unrecognized conv: {conv} [options: gat, gatv2, transformer, gine, nnconv]')

        convs = [] 
        for i in range(layers): 
            convs.append(conv())
        self.convs = torch.nn.ModuleList(convs)    

        self.norm = pyg.nn.norm.LayerNorm(channels, affine=norm_affine, mode=norm_mode)

        self.edge_embedding = torch.nn.ParameterList( [torch.nn.Embedding(n_relations, edge_dim)  for i in range(layers)] )

        self.nonlin = nonlin() 

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(channels, 100),
            nonlin(),
            torch.nn.Linear(100, 1)
        )
    
    def forward(self, x, edge_index, edge_type):
        
        # x is shape: (N, 1)
        x = x.expand(-1, self.channels)

        for i,conv in enumerate(self.convs):
            xl = x
            edge_attr = self.edge_embedding[i](edge_type)
            if self.checkpoint: 
                x = torch.utils.checkpoint.checkpoint(conv, x, edge_index, edge_attr, use_reentrant=False)
            else: 
                x = conv(x=x, edge_index=edge_index, edge_attr=edge_attr)
            if self.residual and (i > 0): x = x + xl
            x = self.norm(x)
            x = self.nonlin(x)

        out = self.mlp(x)
        out = out.sigmoid()
        return out 