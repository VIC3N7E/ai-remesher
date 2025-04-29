import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from typing import Tuple, Optional, Dict, List

class FeatureExtractor(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, hidden_channels: int, num_heads: int = 8):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_channels, num_heads)
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(hidden_channels)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 4),
            nn.ReLU(),
            nn.Linear(hidden_channels * 4, hidden_channels)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x

class TopologyPredictor(nn.Module):
    def __init__(self, hidden_channels: int):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.predictor(x)

class RetopologyModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 128,
        num_transformer_layers: int = 4
    ):
        super().__init__()
        
        # Feature extraction
        self.feature_extractor = FeatureExtractor(in_channels, hidden_channels)
        
        # Transformer layers for global context
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(hidden_channels)
            for _ in range(num_transformer_layers)
        ])
        
        # Topology prediction
        self.topology_predictor = TopologyPredictor(hidden_channels)
        
        # UV and normal preservation
        self.attribute_predictor = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 6)  # 3 for normals, 3 for UV coordinates
        )
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Extract local features
        local_features = self.feature_extractor(x, edge_index)
        
        # Apply transformer layers for global context
        global_features = local_features
        for transformer in self.transformer_layers:
            global_features = transformer(global_features)
        
        # Predict topology
        topology_logits = self.topology_predictor(global_features)
        
        # Predict attributes (normals and UVs)
        attributes = self.attribute_predictor(global_features)
        normals = attributes[:, :3]
        uvs = attributes[:, 3:]
        
        return {
            'topology_logits': topology_logits,
            'normals': normals,
            'uvs': uvs,
            'features': global_features
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