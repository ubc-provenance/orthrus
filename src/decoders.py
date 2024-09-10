import torch
import torch.nn as nn

class EdgeTypeDecoder(nn.Module):
    def __init__(self, in_dim, num_edge_types, loss_fn, dropout, num_layers, activation):
        super(EdgeTypeDecoder, self).__init__()
        self.lin_src = nn.Linear(in_dim, in_dim*2)
        self.lin_dst = nn.Linear(in_dim, in_dim*2)
        
        if num_layers == 2:
            self.lin_seq = nn.Sequential(
                nn.Linear(in_dim * 4, in_dim * 2),
                nn.Dropout(dropout),
                activation,
                
                nn.Linear(in_dim * 2, num_edge_types),
            )
        elif num_layers == 3:
            self.lin_seq = nn.Sequential(
                nn.Linear(in_dim * 4, in_dim * 4),
                nn.Dropout(dropout),
                activation,
            
                nn.Linear(in_dim * 4, in_dim * 2),
                nn.Dropout(dropout),
                activation,
            
                nn.Linear(in_dim * 2, num_edge_types),
            )
        else:
            raise ValueError(f"Invalid number of layers, found {num_layers}")
        
        self.loss_fn = loss_fn
        
    def forward(self, h_src, h_dst, edge_type, inference, **kwargs):
        h = torch.cat([self.lin_src(h_src), self.lin_dst(h_dst)], dim=-1)
        h = self.lin_seq(h)
        
        edge_type_classes = edge_type.argmax(dim=1)
        loss = self.loss_fn(h, edge_type_classes, inference=inference)
        return loss
