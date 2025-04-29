"""
Feature extraction for 3D meshes.

This module implements the FeatureExtractor class, which is responsible for
extracting various features from 3D meshes, including geometric features,
topological features, and curvature-based features.
"""

import numpy as np
import trimesh
import pyvista as pv
from typing import Dict, List, Tuple, Union, Optional, Any
from pathlib import Path
from collections import defaultdict


class FeatureExtractor:
    """
    A class for extracting features from 3D meshes.
    
    This class provides methods for extracting various types of features
    from 3D meshes, including:
    - Geometric features (vertex positions, face normals, etc.)
    - Topological features (vertex connectivity, face adjacency, etc.)
    - Curvature-based features (mean curvature, Gaussian curvature, etc.)
    - Feature detection (sharp edges, corners, creases)
    """
    
    def __init__(
        self,
        sharp_angle_threshold: float = 30.0,
        corner_threshold: int = 2,
        crease_threshold: float = 45.0
    ):
        """
        Initialize the feature extractor.
        
        Args:
            sharp_angle_threshold: Angle threshold in degrees for sharp edge detection
            corner_threshold: Number of sharp edges required to form a corner
            crease_threshold: Angle threshold in degrees for crease detection
        """
        self.sharp_angle_threshold = np.radians(sharp_angle_threshold)
        self.corner_threshold = corner_threshold
        self.crease_threshold = np.radians(crease_threshold)
        self.config = {}
        self.mesh = None
        self.trimesh_mesh = None
        self.pv_mesh = None
        
    def load_mesh(self, mesh_path: Union[str, Path]) -> None:
        """
        Load a mesh from a file.
        
        Args:
            mesh_path: Path to the mesh file
        """
        mesh_path = Path(mesh_path)
        if not mesh_path.exists():
            raise FileNotFoundError(f"Mesh file not found: {mesh_path}")
            
        # Load mesh using trimesh
        self.trimesh_mesh = trimesh.load(str(mesh_path))
        
        # Convert to PyVista mesh for some operations
        vertices = np.array(self.trimesh_mesh.vertices)
        faces = np.array(self.trimesh_mesh.faces)
        # PyVista requires faces to be padded with the number of vertices
        faces_padded = np.column_stack((np.full(len(faces), 3), faces))
        self.pv_mesh = pv.PolyData(vertices, faces_padded.flatten())
        
        # Store the original mesh path
        self.mesh_path = mesh_path
        
    def extract_vertex_features(self) -> Dict[str, np.ndarray]:
        """
        Extract vertex-based features from the mesh.
        
        Returns:
            Dictionary of vertex features
        """
        if self.trimesh_mesh is None:
            raise ValueError("No mesh loaded. Call load_mesh() first.")
            
        features = {}
        
        # Vertex positions
        features['positions'] = np.array(self.trimesh_mesh.vertices)
        
        # Vertex normals
        features['normals'] = np.array(self.trimesh_mesh.vertex_normals)
        
        # Vertex curvatures (if available)
        if hasattr(self.trimesh_mesh, 'vertex_defects'):
            features['vertex_defects'] = np.array(self.trimesh_mesh.vertex_defects)
            
        return features
        
    def extract_face_features(self) -> Dict[str, np.ndarray]:
        """
        Extract face-based features from the mesh.
        
        Returns:
            Dictionary of face features
        """
        if self.trimesh_mesh is None:
            raise ValueError("No mesh loaded. Call load_mesh() first.")
            
        features = {}
        
        # Face normals
        features['normals'] = np.array(self.trimesh_mesh.face_normals)
        
        # Face areas
        features['areas'] = np.array(self.trimesh_mesh.area_faces)
        
        # Face centroids
        features['centroids'] = np.array(self.trimesh_mesh.triangles.mean(axis=1))
        
        return features
        
    def compute_curvatures(self) -> Dict[str, np.ndarray]:
        """
        Compute curvature-based features for the mesh.
        
        Returns:
            Dictionary of curvature features
        """
        if self.pv_mesh is None:
            raise ValueError("No mesh loaded. Call load_mesh() first.")
            
        features = {}
        
        # Compute curvatures using PyVista
        # Note: PyVista's curvature method doesn't accept a method parameter anymore
        # It computes mean curvature by default
        curvatures = self.pv_mesh.curvature()
        features['mean_curvature'] = curvatures
        
        # Compute Gaussian curvature using trimesh
        if self.trimesh_mesh is not None:
            # Compute vertex normals if not already computed
            if not hasattr(self.trimesh_mesh, 'vertex_normals') or self.trimesh_mesh.vertex_normals is None:
                self.trimesh_mesh.vertex_normals = self.trimesh_mesh.vertex_normals
                
            # Compute face normals if not already computed
            if not hasattr(self.trimesh_mesh, 'face_normals') or self.trimesh_mesh.face_normals is None:
                self.trimesh_mesh.face_normals = self.trimesh_mesh.face_normals
                
            # Compute Gaussian curvature
            gaussian_curvature = self.trimesh_mesh.vertex_defects
            features['gaussian_curvature'] = gaussian_curvature
        
        return features
        
    def detect_features(self, mesh: trimesh.Trimesh) -> Dict[str, Any]:
        """Detect features in a mesh.
        
        Args:
            mesh: The mesh to detect features in
            
        Returns:
            Dictionary containing detected features:
            - sharp_edges: List of edge indices
            - corners: List of vertex indices
            - creases: List of edge indices
        """
        self.trimesh_mesh = mesh
        
        # Initialize feature containers
        sharp_edges = []
        corners = defaultdict(int)  # vertex index -> count of sharp edges
        
        # Get face adjacency information
        face_adjacency = self.trimesh_mesh.face_adjacency
        face_adjacency_angles = self.trimesh_mesh.face_adjacency_angles
        
        # Detect sharp edges based on dihedral angle
        for i, angle in enumerate(face_adjacency_angles):
            if abs(angle) > self.sharp_angle_threshold:
                edge_vertices = self.trimesh_mesh.face_adjacency_edges[i]
                sharp_edges.append(edge_vertices)
                
                # Count sharp edges per vertex
                corners[edge_vertices[0]] += 1
                corners[edge_vertices[1]] += 1
        
        # Convert sharp edges to numpy array
        sharp_edges = np.array(sharp_edges)
        
        # Detect corners (vertices with multiple sharp edges)
        corner_vertices = [v for v, count in corners.items() if count >= self.corner_threshold]
        
        # Detect creases (connected sharp edges)
        creases = self._detect_creases(sharp_edges)
        
        return {
            'sharp_edges': sharp_edges,
            'corners': np.array(corner_vertices),
            'creases': creases
        }
        
    def extract_all_features(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Extract all available features from the mesh.
        
        Returns:
            Dictionary containing all extracted features
        """
        if self.trimesh_mesh is None:
            raise ValueError("No mesh loaded. Call load_mesh() first.")
            
        all_features = {
            'vertex_features': self.extract_vertex_features(),
            'face_features': self.extract_face_features(),
            'curvatures': self.compute_curvatures(),
            'detected_features': self.detect_features()
        }
        
        return all_features 

    def _detect_creases(self, sharp_edges: np.ndarray) -> List[np.ndarray]:
        """Detect creases (connected sharp edges) in the mesh.
        
        Args:
            sharp_edges: Array of sharp edge vertex pairs
            
        Returns:
            List of arrays containing vertex indices for each crease
        """
        if len(sharp_edges) == 0:
            return []
            
        # Create an adjacency dictionary for sharp edges
        edge_adjacency = defaultdict(list)
        for i, (v1, v2) in enumerate(sharp_edges):
            edge_adjacency[v1].append((i, v2))
            edge_adjacency[v2].append((i, v1))
        
        # Find creases by following connected sharp edges
        creases = []
        visited_edges = set()
        
        for edge_idx in range(len(sharp_edges)):
            if edge_idx in visited_edges:
                continue
                
            # Start a new crease
            crease = []
            stack = [(edge_idx, sharp_edges[edge_idx][0], sharp_edges[edge_idx][1])]
            
            while stack:
                current_edge_idx, v1, v2 = stack.pop()
                
                if current_edge_idx in visited_edges:
                    continue
                    
                visited_edges.add(current_edge_idx)
                crease.extend([v1, v2])
                
                # Check connected edges at v2
                for next_edge_idx, next_v in edge_adjacency[v2]:
                    if next_edge_idx not in visited_edges:
                        # Check if the angle between edges is within threshold
                        if self._check_edge_angle(v1, v2, next_v):
                            stack.append((next_edge_idx, v2, next_v))
            
            if len(crease) > 0:
                # Remove duplicates and convert to numpy array
                crease = np.array(list(dict.fromkeys(crease)))
                creases.append(crease)
        
        return creases

    def _check_edge_angle(self, v1: int, v2: int, v3: int) -> bool:
        """Check if the angle between two edges is within the crease threshold.
        
        Args:
            v1: First vertex index
            v2: Middle vertex index (shared)
            v3: Third vertex index
            
        Returns:
            True if the angle is within threshold, False otherwise
        """
        # Get vertex positions
        p1 = self.trimesh_mesh.vertices[v1]
        p2 = self.trimesh_mesh.vertices[v2]
        p3 = self.trimesh_mesh.vertices[v3]
        
        # Calculate vectors
        vec1 = p1 - p2
        vec2 = p3 - p2
        
        # Calculate angle between vectors
        angle = np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
        
        return abs(angle) < self.crease_threshold 