from provnet_utils import *
from config import *
import torch.nn as nn
from torch_geometric.nn import SAGEConv, TransformerConv


class GraphTransformer(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, edge_dim, dropout, activation, num_heads):
        super(GraphTransformer, self).__init__()
        
        self.conv = TransformerConv(in_dim, hid_dim, heads=num_heads, dropout=dropout, edge_dim=edge_dim)
        self.conv2 = TransformerConv(hid_dim * num_heads, out_dim, heads=1, concat=False, dropout=dropout, edge_dim=edge_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, x, edge_index, edge_feats=None, **kwargs):
        x = self.activation(self.conv(x, edge_index, edge_feats))
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_feats)
        return x
    
class OrthrusEncoder(nn.Module):
    def __init__(
        self,
        encoder,
        neighbor_loader,
        in_dim,
        temporal_dim,
        use_node_feats_in_gnn,
        graph_reindexer,
        edge_features,
        device,
        num_nodes,
        edge_dim,
    ):
        super(OrthrusEncoder, self).__init__()
        self.encoder = encoder
        self.neighbor_loader = neighbor_loader
        self.device = device
        self.assoc = torch.empty(num_nodes, dtype=torch.long, device=device)
        
        self.edge_features = edge_features

        self.use_node_feats_in_gnn = use_node_feats_in_gnn
        if self.use_node_feats_in_gnn:
            self.src_linear = nn.Linear(in_dim, temporal_dim)
            self.dst_linear = nn.Linear(in_dim, temporal_dim)
            
        self.graph_reindexer = graph_reindexer

    def forward(self, edge_index, t, msg, x, full_data, inference=False, **kwargs):
        # NOTE: full_data is the full list of all edges in the entire dataset (train/val/test)
        
        src, dst = edge_index
        x_src, x_dst = x
        batch_edge_index = edge_index.clone()
        
        n_id = torch.cat([src, dst]).unique()
        n_id, edge_index, e_id = self.neighbor_loader(n_id)
        self.assoc[n_id] = torch.arange(n_id.size(0), device=self.device)

        x_proj = None
        x_src, x_dst = self.graph_reindexer.node_features_reshape(batch_edge_index, x_src, x_dst, max_num_node=n_id.max())
        x_proj = self.src_linear(x_src[n_id]) + self.dst_linear(x_dst[n_id])

        h = x_proj
        
        # Edge features
        edge_feats = []
        if "edge_type" in self.edge_features:
            curr_msg = full_data.edge_type[e_id.cpu()].to(self.device)
            edge_feats.append(curr_msg)
        if "msg" in self.edge_features:
            curr_msg = full_data.msg[e_id.cpu()].to(self.device)
            edge_feats.append(curr_msg)
        edge_feats = torch.cat(edge_feats, dim=-1) if len(edge_feats) > 0 else None
        
        h = self.encoder(h, edge_index, edge_feats=edge_feats)

        h_src = h[self.assoc[src]]
        h_dst = h[self.assoc[dst]]

        self.neighbor_loader.insert(src, dst)
        
        return h_src, h_dst

    def reset_state(self):
        self.neighbor_loader.reset_state()
