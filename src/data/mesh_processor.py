import torch
import numpy as np
import trimesh
import open3d as o3d
from scipy.spatial import cKDTree
from typing import Tuple, Dict, List, Optional, Union
from pathlib import Path

class MeshProcessor:
    def __init__(self, config: Dict):
        """
        Initialize the mesh processor with configuration.
        
        Args:
            config (Dict): Configuration dictionary with processing parameters
        """
        self.config = config
        self.max_vertices = config.get('max_vertices', 10000)
        self.normalize = config.get('normalize', True)
        self.compute_normals = config.get('compute_normals', True)
        self.should_compute_curvature = config.get('compute_curvature', True)
        self.detect_features = config.get('detect_features', True)
        self.angle_threshold = config.get('angle_threshold', 30)
        
    def process_mesh(self, mesh: Union[trimesh.Trimesh, str]) -> Dict:
        """
        Process a mesh to extract features and prepare it for the model.
        
        Args:
            mesh: Either a trimesh.Trimesh object or a path to a mesh file
            
        Returns:
            Dict: Dictionary containing processed mesh data and features
        """
        # Load mesh if path is provided
        if isinstance(mesh, str):
            mesh = trimesh.load(mesh)
        
        # Basic mesh validation
        if not mesh.is_watertight:
            mesh.fill_holes()
        
        # Normalize mesh if required
        if self.normalize:
            mesh, transform_params = self.normalize_mesh(mesh)
        else:
            transform_params = {'centroid': np.zeros(3), 'scale': 1.0}
        
        # Extract features
        vertex_features = self.extract_vertex_features(mesh)
        edge_features = self.extract_edge_features(mesh)
        face_features = self.extract_face_features(mesh)
        
        # Convert to tensors
        vertex_features = {k: torch.from_numpy(v).float() for k, v in vertex_features.items()}
        edge_features = {k: torch.from_numpy(v).float() for k, v in edge_features.items()}
        face_features = {k: torch.from_numpy(v).float() for k, v in face_features.items()}
        
        return {
            'vertex_features': vertex_features,
            'edge_features': edge_features,
            'face_features': face_features,
            'transform_params': transform_params,
            'faces': torch.from_numpy(mesh.faces).long()
        }
    
    def normalize_mesh(self, mesh: trimesh.Trimesh) -> Tuple[trimesh.Trimesh, Dict]:
        """
        Normalize mesh to unit sphere.
        
        Args:
            mesh: Input mesh
            
        Returns:
            Tuple[trimesh.Trimesh, Dict]: Normalized mesh and transformation parameters
        """
        # Center mesh
        centroid = mesh.vertices.mean(axis=0)
        mesh.vertices -= centroid
        
        # Scale to unit sphere
        scale = np.max(np.linalg.norm(mesh.vertices, axis=1))
        mesh.vertices /= scale
        
        return mesh, {'centroid': centroid, 'scale': scale}
    
    def extract_vertex_features(self, mesh: trimesh.Trimesh) -> Dict[str, np.ndarray]:
        """
        Extract vertex-level features.
        
        Args:
            mesh: Input mesh
            
        Returns:
            Dict[str, np.ndarray]: Dictionary of vertex features
        """
        features = {
            'positions': mesh.vertices.copy()
        }
        
        # Compute vertex normals if required
        if self.compute_normals:
            if not hasattr(mesh, 'vertex_normals') or mesh.vertex_normals is None:
                mesh.vertex_normals = mesh.face_normals[mesh.vertex_faces].mean(axis=1)
            features['normals'] = mesh.vertex_normals
        
        # Compute curvature if required
        if self.should_compute_curvature:
            features['curvatures'] = self.compute_curvature(mesh)
        
        return features
    
    def extract_edge_features(self, mesh: trimesh.Trimesh) -> Dict[str, np.ndarray]:
        """
        Extract edge-level features.
        
        Args:
            mesh: Input mesh
            
        Returns:
            Dict[str, np.ndarray]: Dictionary of edge features
        """
        # Get unique edges
        edges = mesh.edges_unique
        
        # Compute edge lengths
        edge_lengths = np.linalg.norm(
            mesh.vertices[edges[:, 0]] - mesh.vertices[edges[:, 1]],
            axis=1
        )
        
        # Detect feature lines if required
        if self.detect_features:
            feature_lines = self.detect_feature_lines(mesh)
        else:
            feature_lines = np.array([])
        
        return {
            'lengths': edge_lengths,
            'feature_lines': feature_lines
        }
    
    def extract_face_features(self, mesh: trimesh.Trimesh) -> Dict[str, np.ndarray]:
        """
        Extract face-level features.
        
        Args:
            mesh: Input mesh
            
        Returns:
            Dict[str, np.ndarray]: Dictionary of face features
        """
        # Compute face areas
        face_areas = mesh.area_faces
        
        # Compute face normals
        face_normals = mesh.face_normals
        
        return {
            'areas': face_areas,
            'normals': face_normals
        }
    
    def compute_curvature(self, mesh: trimesh.Trimesh) -> np.ndarray:
        """
        Compute approximate mean curvature using Laplacian smoothing.
        
        Args:
            mesh: Input mesh
            
        Returns:
            np.ndarray: Array of curvature values
        """
        # Get vertices and faces
        vertices = mesh.vertices
        faces = mesh.faces
        
        # Create vertex adjacency matrix
        n_vertices = len(vertices)
        adj_matrix = np.zeros((n_vertices, n_vertices))
        
        # Fill adjacency matrix
        for face in faces:
            for i in range(3):
                v1, v2 = face[i], face[(i + 1) % 3]
                adj_matrix[v1, v2] = 1
                adj_matrix[v2, v1] = 1
        
        # Compute vertex degrees
        degrees = np.sum(adj_matrix, axis=1)
        
        # Compute Laplacian
        laplacian = np.zeros((n_vertices, 3))
        for i in range(n_vertices):
            # Get neighbors
            neighbors = np.where(adj_matrix[i] > 0)[0]
            if len(neighbors) > 0:
                # Compute mean position of neighbors
                mean_pos = np.mean(vertices[neighbors], axis=0)
                # Laplacian is difference between vertex and mean of neighbors
                laplacian[i] = vertices[i] - mean_pos
        
        # Compute curvature as magnitude of Laplacian
        curvature = np.linalg.norm(laplacian, axis=1)
        
        # Normalize curvature
        curvature = (curvature - np.min(curvature)) / (np.max(curvature) - np.min(curvature) + 1e-6)
        
        return curvature
    
    def detect_feature_lines(self, mesh: trimesh.Trimesh) -> np.ndarray:
        """
        Detect sharp edges based on dihedral angles.
        
        Args:
            mesh: Input mesh
            
        Returns:
            np.ndarray: Array of feature edge indices
        """
        # Get edges and their adjacent faces
        edges = mesh.edges_unique
        edges_face = mesh.edges_face
        
        # Initialize feature edges array
        feature_edges = np.zeros(len(edges), dtype=bool)
        
        # Compute dihedral angles for each edge
        for i, (edge, faces) in enumerate(zip(edges, edges_face)):
            # Convert faces to array if it's a single integer
            if isinstance(faces, (int, np.int64)):
                faces = np.array([faces])
            elif isinstance(faces, list):
                faces = np.array(faces)
            
            # Handle boundary edges and edges with insufficient faces
            if len(faces) < 2 or -1 in faces:
                feature_edges[i] = True
                continue
            
            # Get face normals
            normal1 = mesh.face_normals[faces[0]]
            normal2 = mesh.face_normals[faces[1]]
            
            # Compute angle between normals
            angle = np.arccos(np.clip(np.dot(normal1, normal2), -1.0, 1.0))
            angle_deg = np.degrees(angle)
            
            # Mark as feature if angle exceeds threshold
            if angle_deg > self.angle_threshold:
                feature_edges[i] = True
        
        return feature_edges 