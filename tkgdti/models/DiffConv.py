import torch
from torch import nn
from torch_geometric.nn import MessagePassing

class DiffConv(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_channels, bias=True):
        super().__init__(aggr='add')
        self.bilin = nn.Bilinear(in_channels, edge_channels, out_channels, bias=bias)

    def forward(self, x, edge_index, edge_attr):

        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        # x_i is the aggregator 
        # x_j is the sender

        msg = self.bilin(x_j, edge_attr)

        ## could add attention here instead of max 

        return msg

    def update(self, aggr_out):
        # Final update step
        return aggr_out