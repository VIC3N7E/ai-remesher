import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from typing import Dict, Optional

class SimpleRetopologyModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 6,
        hidden_channels: int = 32
    ):
        super().__init__()
        
        # Feature extraction with GCN layers
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        
        # Prediction heads
        self.topology_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1)
        )
        
        self.attribute_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 6)  # 3 for normals, 3 for UV coordinates
        )
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Extract features
        x = F.relu(self.conv1(x, edge_index))
        features = self.conv2(x, edge_index)
        
        # Predict topology
        topology_logits = self.topology_head(features)
        
        # Predict attributes
        attributes = self.attribute_head(features)
        normals = attributes[:, :3]
        uvs = attributes[:, 3:]
        
        return {
            'topology_logits': topology_logits,
            'normals': normals,
            'uvs': uvs,
            'features': features
        }
    
    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        # Topology loss
        topology_loss = F.binary_cross_entropy_with_logits(
            predictions['topology_logits'],
            targets['topology']
        )
        
        # Normal consistency loss
        normal_loss = F.mse_loss(
            predictions['normals'],
            targets['normals']
        )
        
        # UV preservation loss
        uv_loss = F.mse_loss(
            predictions['uvs'],
            targets['uvs']
        )
        
        # Feature preservation loss
        feature_loss = F.mse_loss(
            predictions['features'],
            targets['features']
        )
        
        total_loss = (
            topology_loss +
            0.1 * normal_loss +
            0.1 * uv_loss +
            0.05 * feature_loss
        )
        
        return {
            'total_loss': total_loss,
            'topology_loss': topology_loss,
            'normal_loss': normal_loss,
            'uv_loss': uv_loss,
            'feature_loss': feature_loss
        } 