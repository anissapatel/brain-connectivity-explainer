import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class BrainGNN(nn.Module):
    """Configurable GNN for fixed-atlas brain graphs.

    feat:    'corrz' = (standardised) correlation profile as node features;
             'id'    = learnable ROI embedding only -> all subject info lives in the edges.
    readout: 'mean' (permutation invariant) or 'flatten' (keeps ROI identity).
    conv:    'gcn' (sym-normalised, needs nonnegative weights) or 'graph' (GraphConv, signed ok).
    """

    def __init__(self, feat="corrz", readout="flatten", conv="gcn", hidden=64,
                 dropout=0.5, n_roi=116, out_ch=8):
        super().__init__()
        from torch_geometric.nn import GraphConv
        self.feat, self.readout, self.R, self.out_ch, self.dropout = feat, readout, n_roi, out_ch, dropout
        in_dim = n_roi
        if feat == "id":
            self.emb = nn.Embedding(n_roi, hidden)
            in_dim = hidden
        Conv = GCNConv if conv == "gcn" else GraphConv
        self.c1, self.c2 = Conv(in_dim, hidden), Conv(hidden, hidden)
        self.bn1, self.bn2 = nn.BatchNorm1d(hidden), nn.BatchNorm1d(hidden)
        if readout == "mean":
            self.head = nn.Linear(hidden, 1)
        else:
            self.proj = nn.Linear(hidden, out_ch)
            self.head = nn.Linear(n_roi * out_ch, 1)

    def forward(self, x, edge_index, batch=None, edge_weight=None, edge_z=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        h = x
        if self.feat == "id":
            h = self.emb(torch.arange(x.size(0), device=x.device) % self.R)
        h = F.relu(self.bn1(self.c1(h, edge_index, edge_weight)))
        h = F.dropout(h, self.dropout, self.training)
        h = F.relu(self.bn2(self.c2(h, edge_index, edge_weight)))
        if self.readout == "mean":
            g = global_mean_pool(h, batch)
        else:  # graphs are batched consecutively with exactly R nodes each
            g = F.relu(self.proj(h)).reshape(-1, self.R * self.out_ch)
        g = F.dropout(g, self.dropout, self.training)
        return self.head(g)


from torch_geometric.nn import MessagePassing, global_add_pool


class EdgeLinear(MessagePassing):
    """Linear model over the graph's edges: sum_e W[roi_u, roi_v] * z_e (W symmetric).

    Written as message passing so GNNExplainer's edge mask also gates this path;
    removing an edge removes its contribution.
    """

    def __init__(self, n_roi=116):
        super().__init__(aggr="add")
        self.R = n_roi
        self.W = nn.Parameter(torch.zeros(n_roi, n_roi))

    def forward(self, edge_index, edge_z, n_nodes, batch):
        roi = torch.arange(n_nodes, device=edge_index.device) % self.R
        out = self.propagate(edge_index, z=edge_z.view(-1, 1), ri=roi.view(-1, 1), size=(n_nodes, n_nodes))
        return global_add_pool(out, batch)  # [B, 1]

    def message(self, z, ri_i, ri_j):
        Ws = 0.5 * (self.W + self.W.t())
        return z * Ws[ri_j.view(-1), ri_i.view(-1)].view(-1, 1)


class HybridGNN(nn.Module):
    """logit = bias + EdgeLinear(graph edges) [+ edge-only GNN]."""

    def __init__(self, use_gnn=True, n_roi=116, **gnn_kw):
        super().__init__()
        self.lin = EdgeLinear(n_roi)
        self.bias = nn.Parameter(torch.zeros(1))
        self.gnn = BrainGNN(n_roi=n_roi, **gnn_kw) if use_gnn else None

    def forward(self, x, edge_index, batch=None, edge_weight=None, edge_z=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        out = self.lin(edge_index, edge_z, x.size(0), batch) + self.bias
        if self.gnn is not None:
            out = out + self.gnn(x, edge_index, batch, edge_weight)
        return out
