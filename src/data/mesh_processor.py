import torch
import numpy as np
import trimesh
from typing import Tuple, Dict, List, Optional
from pathlib import Path

class MeshProcessor:
    def __init__(
        self,
        max_vertices: int = 10000,
        normalize: bool = True,
        compute_normals: bool = True,
        compute_curvature: bool = True,
        detect_features: bool = True
    ):
        self.max_vertices = max_vertices
        self.normalize = normalize
        self.compute_normals = compute_normals
        self.compute_curvature = compute_curvature
        self.detect_features = detect_features
        
    def process_mesh(self, mesh: trimesh.Trimesh) -> Dict[str, torch.Tensor]:
        """Process a mesh and return tensors for model input."""
        # Basic mesh processing
        if self.normalize:
            mesh = self._normalize_mesh(mesh)
            
        # Compute vertex features
        vertex_features = self._compute_vertex_features(mesh)
        
        # Compute edge features
        edge_index, edge_features = self._compute_edge_features(mesh)
        
        # Compute face features
        face_features = self._compute_face_features(mesh)
        
        return {
            'vertices': torch.FloatTensor(mesh.vertices),
            'faces': torch.LongTensor(mesh.faces),
            'vertex_features': torch.FloatTensor(vertex_features),
            'edge_index': torch.LongTensor(edge_index),
            'edge_features': torch.FloatTensor(edge_features),
            'face_features': torch.FloatTensor(face_features)
        }
    
    def _normalize_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Normalize mesh vertices to unit sphere."""
        vertices = mesh.vertices
        center = vertices.mean(axis=0)
        vertices = vertices - center
        scale = np.abs(vertices).max()
        vertices = vertices / scale
        return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)
    
    def _compute_vertex_features(
        self,
        mesh: trimesh.Trimesh
    ) -> np.ndarray:
        """Compute vertex features including position, normal, and curvature."""
        features = []
        
        # Position
        features.append(mesh.vertices)
        
        # Normals
        if self.compute_normals:
            vertex_normals = mesh.vertex_normals
            features.append(vertex_normals)
        
        # Curvature
        if self.compute_curvature:
            curvature = self._compute_curvature(mesh)
            features.append(curvature)
        
        # Feature detection
        if self.detect_features:
            feature_mask = self._detect_features(mesh)
            features.append(feature_mask)
        
        return np.concatenate(features, axis=1)
    
    def _compute_edge_features(
        self,
        mesh: trimesh.Trimesh
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute edge features and connectivity."""
        # Get edge connectivity
        edges = mesh.edges_unique
        edge_index = edges.T
        
        # Compute edge features
        edge_features = []
        
        # Edge length
        edge_lengths = np.linalg.norm(
            mesh.vertices[edges[:, 0]] - mesh.vertices[edges[:, 1]],
            axis=1
        )
        edge_features.append(edge_lengths)
        
        # Dihedral angle
        dihedral_angles = self._compute_dihedral_angles(mesh, edges)
        edge_features.append(dihedral_angles)
        
        return edge_index, np.stack(edge_features, axis=1)
    
    def _compute_face_features(
        self,
        mesh: trimesh.Trimesh
    ) -> np.ndarray:
        """Compute face features."""
        features = []
        
        # Face normals
        face_normals = mesh.face_normals
        features.append(face_normals)
        
        # Face area
        face_areas = mesh.area_faces
        features.append(face_areas.reshape(-1, 1))
        
        return np.concatenate(features, axis=1)
    
    def _compute_curvature(self, mesh: trimesh.Trimesh) -> np.ndarray:
        """Compute mean curvature at each vertex."""
        # Simple curvature estimation using vertex normals
        vertex_normals = mesh.vertex_normals
        curvature = np.zeros(len(mesh.vertices))
        
        for vertex_idx in range(len(mesh.vertices)):
            # Get neighboring vertices
            neighbors = mesh.vertex_neighbors[vertex_idx]
            if len(neighbors) > 0:
                # Compute curvature as variation in normal direction
                neighbor_normals = vertex_normals[neighbors]
                curvature[vertex_idx] = np.mean(
                    np.abs(neighbor_normals - vertex_normals[vertex_idx])
                )
        
        return curvature.reshape(-1, 1)
    
    def _detect_features(self, mesh: trimesh.Trimesh) -> np.ndarray:
        """Detect feature lines using dihedral angles."""
        # Get unique edges
        edges = mesh.edges_unique
        
        # Compute dihedral angles
        dihedral_angles = self._compute_dihedral_angles(mesh, edges)
        
        # Create feature mask
        feature_mask = np.zeros(len(mesh.vertices))
        feature_threshold = np.pi / 4  # 45 degrees
        
        # Mark vertices connected to sharp edges
        sharp_edges = edges[dihedral_angles > feature_threshold]
        feature_mask[sharp_edges.flatten()] = 1
        
        return feature_mask.reshape(-1, 1)
    
    def _compute_dihedral_angles(
        self,
        mesh: trimesh.Trimesh,
        edges: np.ndarray
    ) -> np.ndarray:
        """Compute dihedral angles for edges."""
        angles = np.zeros(len(edges))
        
        for i, edge in enumerate(edges):
            # Get faces sharing the edge
            edge_faces = mesh.face_adjacency_edges == edge
            if edge_faces.any():
                # Get normals of adjacent faces
                face_normals = mesh.face_normals[edge_faces]
                if len(face_normals) == 2:
                    # Compute angle between normals
                    angle = np.arccos(
                        np.clip(
                            np.dot(face_normals[0], face_normals[1]),
                            -1.0,
                            1.0
                        )
                    )
                    angles[i] = angle
        
        return angles 