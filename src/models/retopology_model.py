import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
import numpy as np
import trimesh
from typing import Tuple, List
import torch.optim as optim

class MeshEncoder(nn.Module):
    """Encoder for mesh features."""
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int = 3):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(GCNConv(in_channels, hidden_channels))
        
        # Middle layers
        for _ in range(num_layers - 2):
            self.layers.append(GCNConv(hidden_channels, hidden_channels))
            
        # Last layer
        self.layers.append(GCNConv(hidden_channels, hidden_channels))
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                x = F.relu(layer(x, edge_index))
            else:
                x = layer(x, edge_index)
        return x

class MeshDecoder(nn.Module):
    """Decoder for mesh reconstruction."""
    def __init__(self, hidden_channels: int, out_channels: int, num_layers: int = 3):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(GCNConv(hidden_channels, hidden_channels))
        
        # Middle layers
        for _ in range(num_layers - 2):
            self.layers.append(GCNConv(hidden_channels, hidden_channels))
            
        # Last layer
        self.layers.append(GCNConv(hidden_channels, out_channels))
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                x = F.relu(layer(x, edge_index))
            else:
                x = layer(x, edge_index)
        return x

class RetopologyModel(nn.Module):
    """Model for mesh retopology."""
    def __init__(self, in_channels: int = 3, hidden_channels: int = 128, num_layers: int = 3):
        super().__init__()
        self.encoder = MeshEncoder(in_channels, hidden_channels, num_layers)
        self.decoder = MeshDecoder(hidden_channels, in_channels, num_layers)
        
    def forward(self, data: Data) -> torch.Tensor:
        # Encode mesh features
        x = self.encoder(data.x, data.edge_index)
        
        # Decode to vertex positions
        vertex_positions = self.decoder(x, data.edge_index)
        
        return vertex_positions

def mesh_to_graph(mesh: trimesh.Trimesh) -> Data:
    """Convert a trimesh to a PyTorch Geometric Data object."""
    # Get vertices and faces
    vertices = torch.tensor(mesh.vertices, dtype=torch.float32)
    faces = torch.tensor(mesh.faces, dtype=torch.long)
    
    # Normalize vertices to [-1, 1] range
    vertices = vertices - vertices.mean(dim=0)
    scale = torch.max(torch.abs(vertices))
    vertices = vertices / scale
    
    # Create edge index from faces (optimized version)
    edge_index = torch.stack([
        faces[:, [0, 1, 1, 2, 2, 0]].flatten(),
        faces[:, [1, 0, 2, 1, 0, 2]].flatten()
    ])
    
    # Remove duplicate edges
    edge_index = torch.unique(edge_index, dim=1)
    
    # Create face index tensor
    face_index = faces.t().contiguous()
    
    return Data(
        x=vertices,
        edge_index=edge_index,
        face_index=face_index,
        scale=scale
    )

def compute_chamfer_distance(pred_vertices: torch.Tensor, target_vertices: torch.Tensor) -> torch.Tensor:
    """Compute Chamfer distance between predicted and target vertices."""
    # Use squared distances for better numerical stability
    dist = torch.cdist(pred_vertices, target_vertices, p=2)
    
    # Get minimum distances in both directions
    min_dist_pred_to_target = torch.min(dist, dim=1)[0]
    min_dist_target_to_pred = torch.min(dist, dim=0)[0]
    
    # Compute Chamfer distance
    chamfer_dist = torch.mean(min_dist_pred_to_target) + torch.mean(min_dist_target_to_pred)
    
    return chamfer_dist

def compute_normal_loss(pred_vertices: torch.Tensor, target_vertices: torch.Tensor,
                       pred_faces: torch.Tensor, target_faces: torch.Tensor) -> torch.Tensor:
    """Compute normal consistency loss between predicted and target meshes."""
    # Compute face normals for both meshes
    pred_normals = compute_face_normals(pred_vertices, pred_faces)
    target_normals = compute_face_normals(target_vertices, target_faces)
    
    # Compute cosine similarity between normals
    similarity = torch.sum(pred_normals * target_normals, dim=1)
    normal_loss = 1 - torch.mean(similarity)
    
    return normal_loss

def compute_face_normals(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Compute face normals for a mesh."""
    # Get vertices for each face (optimized version)
    face_vertices = vertices[faces]
    
    # Compute two edge vectors
    edge1 = face_vertices[:, 1] - face_vertices[:, 0]
    edge2 = face_vertices[:, 2] - face_vertices[:, 0]
    
    # Compute normal using cross product (fixing the deprecation warning)
    normals = torch.linalg.cross(edge1, edge2)
    
    # Normalize
    normals = F.normalize(normals, p=2, dim=1)
    
    return normals

class RetopologyTrainer:
    """Trainer for the retopology model."""
    def __init__(self, model: RetopologyModel, learning_rate: float = 0.001):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        self.criterion = nn.MSELoss()
        self.grad_clip_value = 1.0  # Maximum gradient norm
        
    def train_step(self, input_mesh: Data, target_mesh: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform a single training step."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        pred_vertices = self.model(input_mesh)
        
        # Compute losses
        chamfer_loss = compute_chamfer_distance(pred_vertices, target_mesh.x)
        normal_loss = compute_normal_loss(
            pred_vertices, target_mesh.x,
            input_mesh.face_index, target_mesh.face_index
        )
        
        # Combine losses with better weighting
        total_loss = chamfer_loss + 0.01 * normal_loss  # Reduced normal loss weight
        
        # Backward pass
        total_loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_value)
        
        # Update weights
        self.optimizer.step()
        
        # Update learning rate
        self.scheduler.step(total_loss)
        
        return chamfer_loss.item(), normal_loss.item()
    
    def validate(self, input_mesh: Data, target_mesh: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """Validate the model on a single mesh pair."""
        self.model.eval()
        with torch.no_grad():
            pred_vertices = self.model(input_mesh)
            
            chamfer_loss = compute_chamfer_distance(pred_vertices, target_mesh.x)
            normal_loss = compute_normal_loss(
                pred_vertices, target_mesh.x,
                input_mesh.face_index, target_mesh.face_index
            )
            
            return chamfer_loss.item(), normal_loss.item()

def retopologize_mesh(model: RetopologyModel, input_mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Retopologize a mesh using the trained model."""
    model.eval()
    
    # Convert input mesh to graph
    input_data = mesh_to_graph(input_mesh)
    
    # Get prediction
    with torch.no_grad():
        pred_vertices = model(input_data)
    
    # Create new mesh with predicted vertices
    retopologized_mesh = trimesh.Trimesh(
        vertices=pred_vertices.cpu().numpy(),
        faces=input_mesh.faces
    )
    
    return retopologized_mesh 