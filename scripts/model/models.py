"""
Model Architectures for FEDBANK Phase 1 Baseline & Phase 2 Federation.

GCN, GraphSAGE, and GAT implementations conforming to the specification:
- 3 layers, hidden dimension 128.
- LayerNorm only (Federation-ready; NO BatchNorm to ensure federated averaging works).
- GAT has 4 heads (32 dim per head, concatenated = 128; final layer averages or projects).
- Single state_dict with standard linear/conv parameters.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv


class GCN(nn.Module):
    """3-layer Graph Convolutional Network with LayerNorm and Dropout."""
    def __init__(
        self,
        in_channels: int = 48,
        hidden_channels: int = 128,
        out_channels: int = 1,
        num_layers: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.in_norm = nn.LayerNorm(in_channels)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # First layer
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.norms.append(nn.LayerNorm(hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.norms.append(nn.LayerNorm(hidden_channels))

        # Penultimate layer
        self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.norms.append(nn.LayerNorm(hidden_channels))

        # Final classification head
        self.classifier = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.in_norm(x)
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        logits = self.classifier(x).squeeze(-1)
        return logits


class GraphSAGE(nn.Module):
    """3-layer GraphSAGE (mean aggregation) with LayerNorm and Dropout."""
    def __init__(
        self,
        in_channels: int = 48,
        hidden_channels: int = 128,
        out_channels: int = 1,
        num_layers: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.in_norm = nn.LayerNorm(in_channels)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr="mean"))
        self.norms.append(nn.LayerNorm(hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr="mean"))
            self.norms.append(nn.LayerNorm(hidden_channels))

        self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr="mean"))
        self.norms.append(nn.LayerNorm(hidden_channels))

        self.classifier = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.in_norm(x)
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        logits = self.classifier(x).squeeze(-1)
        return logits


class GAT(nn.Module):
    """
    3-layer Graph Attention Network (GAT) with multi-head attention and LayerNorm.
    Default: 4 heads x 32 hidden = 128 dims.
    """
    def __init__(
        self,
        in_channels: int = 48,
        hidden_channels: int = 128,
        out_channels: int = 1,
        heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.in_norm = nn.LayerNorm(in_channels)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.heads = heads
        head_dim = hidden_channels // heads

        # Layer 1
        self.convs.append(GATConv(in_channels, head_dim, heads=heads, concat=True, dropout=dropout))
        self.norms.append(nn.LayerNorm(hidden_channels))

        # Layer 2
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels, head_dim, heads=heads, concat=True, dropout=dropout))
            self.norms.append(nn.LayerNorm(hidden_channels))

        # Layer 3
        self.convs.append(GATConv(hidden_channels, head_dim, heads=heads, concat=True, dropout=dropout))
        self.norms.append(nn.LayerNorm(hidden_channels))

        self.classifier = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list]:
        x = self.in_norm(x)
        attns = []
        for conv, norm in zip(self.convs, self.norms):
            if return_attention_weights:
                x, alpha = conv(x, edge_index, return_attention_weights=True)
                attns.append(alpha)
            else:
                x = conv(x, edge_index)
            x = norm(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        logits = self.classifier(x).squeeze(-1)
        if return_attention_weights:
            return logits, attns
        return logits


class TabularMLP(nn.Module):
    """3-layer MLP baseline with LayerNorm matching GNN hidden capacity."""
    def __init__(
        self,
        in_channels: int = 48,
        hidden_channels: int = 128,
        out_channels: int = 1,
        num_layers: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.in_norm = nn.LayerNorm(in_channels)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.layers.append(nn.Linear(in_channels, hidden_channels))
        self.norms.append(nn.LayerNorm(hidden_channels))

        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_channels, hidden_channels))
            self.norms.append(nn.LayerNorm(hidden_channels))

        self.layers.append(nn.Linear(hidden_channels, hidden_channels))
        self.norms.append(nn.LayerNorm(hidden_channels))

        self.classifier = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_norm(x)
        for layer, norm in zip(self.layers, self.norms):
            x = layer(x)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        logits = self.classifier(x).squeeze(-1)
        return logits
