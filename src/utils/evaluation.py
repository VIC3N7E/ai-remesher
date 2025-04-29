import numpy as np
import torch
import trimesh
import open3d as o3d
from typing import Dict, List, Tuple, Union, Optional
from scipy.spatial import cKDTree
from pathlib import Path

class MeshEvaluator:
    """
    A class for evaluating the quality of retopologized meshes compared to original meshes.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the mesh evaluator.
        
        Args:
            config: Configuration dictionary with evaluation parameters
        """
        self.config = config or {}
        self.metrics = {}
    
    def evaluate(self, original_mesh: Union[trimesh.Trimesh, str], 
                retopologized_mesh: Union[trimesh.Trimesh, str]) -> Dict[str, float]:
        """
        Evaluate the quality of a retopologized mesh compared to the original mesh.
        
        Args:
            original_mesh: Original mesh (trimesh object or path)
            retopologized_mesh: Retopologized mesh (trimesh object or path)
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Load meshes if paths are provided
        if isinstance(original_mesh, str):
            original_mesh = trimesh.load(original_mesh)
        if isinstance(retopologized_mesh, str):
            retopologized_mesh = trimesh.load(retopologized_mesh)
        
        # Compute all metrics
        self.metrics = {}
        
        # Geometry-based metrics
        self.metrics['vertex_count_ratio'] = self.vertex_count_ratio(original_mesh, retopologized_mesh)
        self.metrics['face_count_ratio'] = self.face_count_ratio(original_mesh, retopologized_mesh)
        self.metrics['volume_ratio'] = self.volume_ratio(original_mesh, retopologized_mesh)
        self.metrics['surface_area_ratio'] = self.surface_area_ratio(original_mesh, retopologized_mesh)
        
        # Distance-based metrics
        self.metrics['hausdorff_distance'] = self.hausdorff_distance(original_mesh, retopologized_mesh)
        self.metrics['mean_distance'] = self.mean_distance(original_mesh, retopologized_mesh)
        self.metrics['rms_distance'] = self.rms_distance(original_mesh, retopologized_mesh)
        
        # Feature preservation metrics
        self.metrics['feature_preservation'] = self.feature_preservation(original_mesh, retopologized_mesh)
        self.metrics['sharp_edge_preservation'] = self.sharp_edge_preservation(original_mesh, retopologized_mesh)
        
        # Quality metrics
        self.metrics['min_angle'] = self.min_angle(retopologized_mesh)
        self.metrics['max_angle'] = self.max_angle(retopologized_mesh)
        self.metrics['aspect_ratio'] = self.aspect_ratio(retopologized_mesh)
        
        return self.metrics
    
    def vertex_count_ratio(self, original_mesh: trimesh.Trimesh, 
                          retopologized_mesh: trimesh.Trimesh) -> float:
        """Compute the ratio of vertex counts between retopologized and original mesh."""
        return len(retopologized_mesh.vertices) / len(original_mesh.vertices)
    
    def face_count_ratio(self, original_mesh: trimesh.Trimesh, 
                        retopologized_mesh: trimesh.Trimesh) -> float:
        """Compute the ratio of face counts between retopologized and original mesh."""
        return len(retopologized_mesh.faces) / len(original_mesh.faces)
    
    def volume_ratio(self, original_mesh: trimesh.Trimesh, 
                    retopologized_mesh: trimesh.Trimesh) -> float:
        """Compute the ratio of volumes between retopologized and original mesh."""
        orig_volume = original_mesh.volume
        retopo_volume = retopologized_mesh.volume
        return retopo_volume / orig_volume if orig_volume > 0 else 0.0
    
    def surface_area_ratio(self, original_mesh: trimesh.Trimesh, 
                          retopologized_mesh: trimesh.Trimesh) -> float:
        """Compute the ratio of surface areas between retopologized and original mesh."""
        orig_area = original_mesh.area
        retopo_area = retopologized_mesh.area
        return retopo_area / orig_area if orig_area > 0 else 0.0
    
    def hausdorff_distance(self, original_mesh: trimesh.Trimesh, 
                          retopologized_mesh: trimesh.Trimesh, 
                          samples: int = 10000) -> float:
        """
        Compute the Hausdorff distance between original and retopologized mesh.
        
        The Hausdorff distance is the maximum of the minimum distances from each point
        in one mesh to any point in the other mesh.
        """
        # Sample points from both meshes
        orig_points = original_mesh.sample(samples)
        retopo_points = retopologized_mesh.sample(samples)
        
        # Compute distances from original to retopologized
        orig_to_retopo = self._compute_distances(orig_points, retopo_points)
        
        # Compute distances from retopologized to original
        retopo_to_orig = self._compute_distances(retopo_points, orig_points)
        
        # Hausdorff distance is the maximum of the two
        return max(np.max(orig_to_retopo), np.max(retopo_to_orig))
    
    def mean_distance(self, original_mesh: trimesh.Trimesh, 
                     retopologized_mesh: trimesh.Trimesh, 
                     samples: int = 10000) -> float:
        """Compute the mean distance between original and retopologized mesh."""
        # Sample points from both meshes
        orig_points = original_mesh.sample(samples)
        retopo_points = retopologized_mesh.sample(samples)
        
        # Compute distances from original to retopologized
        orig_to_retopo = self._compute_distances(orig_points, retopo_points)
        
        # Compute distances from retopologized to original
        retopo_to_orig = self._compute_distances(retopo_points, orig_points)
        
        # Mean distance is the average of all distances
        return np.mean(np.concatenate([orig_to_retopo, retopo_to_orig]))
    
    def rms_distance(self, original_mesh: trimesh.Trimesh, 
                    retopologized_mesh: trimesh.Trimesh, 
                    samples: int = 10000) -> float:
        """Compute the root mean square distance between original and retopologized mesh."""
        # Sample points from both meshes
        orig_points = original_mesh.sample(samples)
        retopo_points = retopologized_mesh.sample(samples)
        
        # Compute distances from original to retopologized
        orig_to_retopo = self._compute_distances(orig_points, retopo_points)
        
        # Compute distances from retopologized to original
        retopo_to_orig = self._compute_distances(retopo_points, orig_points)
        
        # RMS distance is the square root of the mean of squared distances
        return np.sqrt(np.mean(np.concatenate([orig_to_retopo, retopo_to_orig]) ** 2))
    
    def _compute_distances(self, points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
        """Compute the minimum distances from each point in points1 to any point in points2."""
        tree = cKDTree(points2)
        distances, _ = tree.query(points1)
        return distances
    
    def feature_preservation(self, original_mesh: trimesh.Trimesh, 
                           retopologized_mesh: trimesh.Trimesh) -> float:
        """
        Compute a score for how well features are preserved in the retopologized mesh.
        
        This is a simplified version that checks if sharp edges are preserved.
        """
        # Detect sharp edges in original mesh
        orig_edges = self._detect_sharp_edges(original_mesh)
        
        # Detect sharp edges in retopologized mesh
        retopo_edges = self._detect_sharp_edges(retopologized_mesh)
        
        # Count how many sharp edges from original are close to sharp edges in retopologized
        preserved_edges = 0
        for orig_edge in orig_edges:
            # Check if any retopologized edge is close to this original edge
            for retopo_edge in retopo_edges:
                if self._edges_are_close(orig_edge, retopo_edge, original_mesh, retopologized_mesh):
                    preserved_edges += 1
                    break
        
        # Return the ratio of preserved edges
        return preserved_edges / len(orig_edges) if len(orig_edges) > 0 else 1.0
    
    def _detect_sharp_edges(self, mesh: trimesh.Trimesh, angle_threshold: float = 60.0) -> List[Tuple[int, int]]:
        """Detect sharp edges in a mesh based on dihedral angle."""
        sharp_edges = []
        
        # Get edges and their adjacent faces
        edges = mesh.edges_unique
        edges_face = mesh.edges_face
        
        # Check each edge
        for i, edge in enumerate(edges):
            # Get adjacent faces
            face_idx = edges_face[i]
            
            # Skip boundary edges
            if -1 in face_idx:
                sharp_edges.append(tuple(edge))
                continue
            
            # Compute dihedral angle
            if len(face_idx) == 2:  # Only for edges with exactly 2 adjacent faces
                normal1 = mesh.face_normals[face_idx[0]]
                normal2 = mesh.face_normals[face_idx[1]]
                
                # Compute angle between normals
                cos_angle = np.dot(normal1, normal2)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Ensure valid range
                angle = np.degrees(np.arccos(cos_angle))
                
                # Check if angle exceeds threshold
                if angle > angle_threshold:
                    sharp_edges.append(tuple(edge))
        
        return sharp_edges
    
    def _edges_are_close(self, edge1: Tuple[int, int], edge2: Tuple[int, int], 
                        mesh1: trimesh.Trimesh, mesh2: trimesh.Trimesh, 
                        threshold: float = 0.01) -> bool:
        """Check if two edges are close to each other."""
        # Get edge vertices
        v1_1 = mesh1.vertices[edge1[0]]
        v1_2 = mesh1.vertices[edge1[1]]
        v2_1 = mesh2.vertices[edge2[0]]
        v2_2 = mesh2.vertices[edge2[1]]
        
        # Compute edge midpoints
        mid1 = (v1_1 + v1_2) / 2
        mid2 = (v2_1 + v2_2) / 2
        
        # Check if midpoints are close
        return np.linalg.norm(mid1 - mid2) < threshold
    
    def sharp_edge_preservation(self, original_mesh: trimesh.Trimesh, 
                              retopologized_mesh: trimesh.Trimesh) -> float:
        """
        Compute a score for how well sharp edges are preserved in the retopologized mesh.
        
        This is similar to feature_preservation but specifically for sharp edges.
        """
        # Detect sharp edges in original mesh
        orig_edges = self._detect_sharp_edges(original_mesh)
        
        # Detect sharp edges in retopologized mesh
        retopo_edges = self._detect_sharp_edges(retopologized_mesh)
        
        # Count how many sharp edges from original are close to sharp edges in retopologized
        preserved_edges = 0
        for orig_edge in orig_edges:
            # Check if any retopologized edge is close to this original edge
            for retopo_edge in retopo_edges:
                if self._edges_are_close(orig_edge, retopo_edge, original_mesh, retopologized_mesh):
                    preserved_edges += 1
                    break
        
        # Return the ratio of preserved edges
        return preserved_edges / len(orig_edges) if len(orig_edges) > 0 else 1.0
    
    def min_angle(self, mesh: trimesh.Trimesh) -> float:
        """Compute the minimum angle in the mesh."""
        # Get face vertices
        faces = mesh.faces
        vertices = mesh.vertices
        
        min_angle = float('inf')
        
        # Check each face
        for face in faces:
            # Get vertices of the face
            v1 = vertices[face[0]]
            v2 = vertices[face[1]]
            v3 = vertices[face[2]]
            
            # Compute vectors
            vec1 = v2 - v1
            vec2 = v3 - v1
            vec3 = v1 - v2
            
            # Normalize vectors
            vec1_norm = np.linalg.norm(vec1)
            vec2_norm = np.linalg.norm(vec2)
            vec3_norm = np.linalg.norm(vec3)
            
            if vec1_norm > 0 and vec2_norm > 0 and vec3_norm > 0:
                # Compute angles
                angle1 = np.degrees(np.arccos(np.clip(np.dot(vec1, vec2) / (vec1_norm * vec2_norm), -1.0, 1.0)))
                angle2 = np.degrees(np.arccos(np.clip(np.dot(-vec1, vec3) / (vec1_norm * vec3_norm), -1.0, 1.0)))
                angle3 = 180 - angle1 - angle2
                
                # Update minimum angle
                min_angle = min(min_angle, angle1, angle2, angle3)
        
        return min_angle if min_angle != float('inf') else 0.0
    
    def max_angle(self, mesh: trimesh.Trimesh) -> float:
        """Compute the maximum angle in the mesh."""
        # Get face vertices
        faces = mesh.faces
        vertices = mesh.vertices
        
        max_angle = 0.0
        
        # Check each face
        for face in faces:
            # Get vertices of the face
            v1 = vertices[face[0]]
            v2 = vertices[face[1]]
            v3 = vertices[face[2]]
            
            # Compute vectors
            vec1 = v2 - v1
            vec2 = v3 - v1
            vec3 = v1 - v2
            
            # Normalize vectors
            vec1_norm = np.linalg.norm(vec1)
            vec2_norm = np.linalg.norm(vec2)
            vec3_norm = np.linalg.norm(vec3)
            
            if vec1_norm > 0 and vec2_norm > 0 and vec3_norm > 0:
                # Compute angles
                angle1 = np.degrees(np.arccos(np.clip(np.dot(vec1, vec2) / (vec1_norm * vec2_norm), -1.0, 1.0)))
                angle2 = np.degrees(np.arccos(np.clip(np.dot(-vec1, vec3) / (vec1_norm * vec3_norm), -1.0, 1.0)))
                angle3 = 180 - angle1 - angle2
                
                # Update maximum angle
                max_angle = max(max_angle, angle1, angle2, angle3)
        
        return max_angle
    
    def aspect_ratio(self, mesh: trimesh.Trimesh) -> float:
        """Compute the average aspect ratio of triangles in the mesh."""
        # Get face vertices
        faces = mesh.faces
        vertices = mesh.vertices
        
        aspect_ratios = []
        
        # Check each face
        for face in faces:
            # Get vertices of the face
            v1 = vertices[face[0]]
            v2 = vertices[face[1]]
            v3 = vertices[face[2]]
            
            # Compute edge lengths
            edge1 = np.linalg.norm(v2 - v1)
            edge2 = np.linalg.norm(v3 - v2)
            edge3 = np.linalg.norm(v1 - v3)
            
            # Compute semi-perimeter
            s = (edge1 + edge2 + edge3) / 2
            
            # Compute area using Heron's formula
            area = np.sqrt(s * (s - edge1) * (s - edge2) * (s - edge3))
            
            # Compute inradius
            inradius = area / s
            
            # Compute circumradius
            circumradius = (edge1 * edge2 * edge3) / (4 * area)
            
            # Compute aspect ratio
            aspect_ratio = circumradius / (2 * inradius)
            
            aspect_ratios.append(aspect_ratio)
        
        # Return average aspect ratio
        return np.mean(aspect_ratios) if aspect_ratios else 0.0
    
    def save_metrics(self, output_path: Union[str, Path]) -> None:
        """Save the computed metrics to a file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            for metric, value in self.metrics.items():
                f.write(f"{metric}: {value}\n")
    
    def print_metrics(self) -> None:
        """Print the computed metrics."""
        print("Mesh Evaluation Metrics:")
        print("-" * 30)
        for metric, value in self.metrics.items():
            print(f"{metric}: {value:.6f}") 