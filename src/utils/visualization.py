import numpy as np
import matplotlib.pyplot as plt
import trimesh
import open3d as o3d
from typing import Dict, List, Tuple, Union, Optional
from pathlib import Path
import os
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

class MeshVisualizer:
    """
    A class for visualizing and comparing original and retopologized meshes.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the mesh visualizer.
        
        Args:
            config: Configuration dictionary with visualization parameters
        """
        self.config = config or {}
        self.output_dir = Path(self.config.get('output_dir', 'output/visualizations'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def visualize_comparison(self, original_mesh: Union[trimesh.Trimesh, str], 
                           retopologized_mesh: Union[trimesh.Trimesh, str], 
                           output_path: Optional[Union[str, Path]] = None,
                           show_distance_map: bool = True,
                           show_feature_preservation: bool = True,
                           show_mesh_stats: bool = True) -> None:
        """
        Visualize a comparison between original and retopologized meshes.
        
        Args:
            original_mesh: Original mesh (trimesh object or path)
            retopologized_mesh: Retopologized mesh (trimesh object or path)
            output_path: Path to save the visualization (optional)
            show_distance_map: Whether to show the distance map
            show_feature_preservation: Whether to show feature preservation
            show_mesh_stats: Whether to show mesh statistics
        """
        # Load meshes if paths are provided
        if isinstance(original_mesh, str):
            original_mesh = trimesh.load(original_mesh)
        if isinstance(retopologized_mesh, str):
            retopologized_mesh = trimesh.load(retopologized_mesh)
        
        # Create output directory if needed
        if output_path is None:
            output_path = self.output_dir / "comparison.png"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create figure with subplots
        num_plots = 1
        if show_distance_map:
            num_plots += 1
        if show_feature_preservation:
            num_plots += 1
        if show_mesh_stats:
            num_plots += 1
        
        fig = plt.figure(figsize=(5 * num_plots, 10))
        
        # Plot original and retopologized meshes
        ax1 = fig.add_subplot(2, num_plots, 1, projection='3d')
        self._plot_mesh(ax1, original_mesh, title="Original Mesh")
        
        ax2 = fig.add_subplot(2, num_plots, num_plots + 1, projection='3d')
        self._plot_mesh(ax2, retopologized_mesh, title="Retopologized Mesh")
        
        # Plot distance map if requested
        if show_distance_map:
            ax3 = fig.add_subplot(2, num_plots, 2, projection='3d')
            self._plot_distance_map(ax3, original_mesh, retopologized_mesh, title="Distance Map")
        
        # Plot feature preservation if requested
        if show_feature_preservation:
            ax4 = fig.add_subplot(2, num_plots, 3 if show_distance_map else 2, projection='3d')
            self._plot_feature_preservation(ax4, original_mesh, retopologized_mesh, title="Feature Preservation")
        
        # Plot mesh statistics if requested
        if show_mesh_stats:
            ax5 = fig.add_subplot(2, num_plots, 4 if show_distance_map and show_feature_preservation else 
                                         3 if (show_distance_map or show_feature_preservation) else 2, projection='3d')
            self._plot_mesh_stats(ax5, original_mesh, retopologized_mesh, title="Mesh Statistics")
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved to {output_path}")
    
    def _plot_mesh(self, ax, mesh: trimesh.Trimesh, title: str = "") -> None:
        """Plot a mesh in a 3D subplot."""
        # Extract vertices and faces
        vertices = mesh.vertices
        faces = mesh.faces
        
        # Plot the mesh
        ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                        triangles=faces, color='lightblue', alpha=0.8)
        
        # Set title and labels
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Set equal aspect ratio
        ax.set_box_aspect([1, 1, 1])
    
    def _plot_distance_map(self, ax, original_mesh: trimesh.Trimesh, 
                          retopologized_mesh: trimesh.Trimesh, title: str = "") -> None:
        """Plot a distance map between original and retopologized meshes."""
        # Sample points from original mesh
        samples = 10000
        orig_points = original_mesh.sample(samples)
        
        # Compute distances from original points to retopologized mesh
        distances = self._compute_point_to_mesh_distances(orig_points, retopologized_mesh)
        
        # Create a scatter plot with color based on distance
        scatter = ax.scatter(orig_points[:, 0], orig_points[:, 1], orig_points[:, 2],
                            c=distances, cmap='viridis', s=1)
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax, label='Distance')
        
        # Set title and labels
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Set equal aspect ratio
        ax.set_box_aspect([1, 1, 1])
    
    def _compute_point_to_mesh_distances(self, points: np.ndarray, mesh: trimesh.Trimesh) -> np.ndarray:
        """Compute distances from points to a mesh."""
        # Create a KD-tree from the mesh vertices
        tree = o3d.geometry.KDTreeFlann(o3d.utility.Vector3dVector(mesh.vertices))
        
        # Compute distances
        distances = np.zeros(len(points))
        for i, point in enumerate(points):
            # Find the nearest neighbor
            [k, idx, dist] = tree.search_knn_vector_3d(point, 1)
            distances[i] = np.sqrt(dist[0])
        
        return distances
    
    def _plot_feature_preservation(self, ax, original_mesh: trimesh.Trimesh, 
                                 retopologized_mesh: trimesh.Trimesh, title: str = "") -> None:
        """Plot feature preservation between original and retopologized meshes."""
        # Detect sharp edges in original mesh
        orig_edges = self._detect_sharp_edges(original_mesh)
        
        # Detect sharp edges in retopologized mesh
        retopo_edges = self._detect_sharp_edges(retopologized_mesh)
        
        # Plot original mesh with sharp edges highlighted
        self._plot_mesh_with_edges(ax, original_mesh, orig_edges, title)
    
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
    
    def _plot_mesh_with_edges(self, ax, mesh: trimesh.Trimesh, 
                            edges: List[Tuple[int, int]], title: str = "") -> None:
        """Plot a mesh with highlighted edges."""
        # Extract vertices and faces
        vertices = mesh.vertices
        faces = mesh.faces
        
        # Plot the mesh
        ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                        triangles=faces, color='lightblue', alpha=0.8)
        
        # Plot the edges
        for edge in edges:
            v1 = vertices[edge[0]]
            v2 = vertices[edge[1]]
            ax.plot([v1[0], v2[0]], [v1[1], v2[1]], [v1[2], v2[2]], 'r-', linewidth=2)
        
        # Set title and labels
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Set equal aspect ratio
        ax.set_box_aspect([1, 1, 1])
    
    def _plot_mesh_stats(self, ax, original_mesh: trimesh.Trimesh, 
                        retopologized_mesh: trimesh.Trimesh, title: str = "") -> None:
        """Plot mesh statistics."""
        # Compute statistics
        orig_vertices = len(original_mesh.vertices)
        orig_faces = len(original_mesh.faces)
        retopo_vertices = len(retopologized_mesh.vertices)
        retopo_faces = len(retopologized_mesh.faces)
        
        # Create bar chart
        labels = ['Vertices', 'Faces']
        orig_values = [orig_vertices, orig_faces]
        retopo_values = [retopo_vertices, retopo_faces]
        
        x = np.arange(len(labels))
        width = 0.35
        
        ax.bar(x - width/2, orig_values, width, label='Original')
        ax.bar(x + width/2, retopo_values, width, label='Retopologized')
        
        # Add labels and title
        ax.set_ylabel('Count')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        
        # Add text annotations
        for i, v in enumerate(orig_values):
            ax.text(i - width/2, v, str(v), ha='center', va='bottom')
        
        for i, v in enumerate(retopo_values):
            ax.text(i + width/2, v, str(v), ha='center', va='bottom')
    
    def visualize_distance_map(self, original_mesh: Union[trimesh.Trimesh, str], 
                              retopologized_mesh: Union[trimesh.Trimesh, str], 
                              output_path: Optional[Union[str, Path]] = None) -> None:
        """
        Visualize the distance map between original and retopologized meshes.
        
        Args:
            original_mesh: Original mesh (trimesh object or path)
            retopologized_mesh: Retopologized mesh (trimesh object or path)
            output_path: Path to save the visualization (optional)
        """
        # Load meshes if paths are provided
        if isinstance(original_mesh, str):
            original_mesh = trimesh.load(original_mesh)
        if isinstance(retopologized_mesh, str):
            retopologized_mesh = trimesh.load(retopologized_mesh)
        
        # Create output directory if needed
        if output_path is None:
            output_path = self.output_dir / "distance_map.png"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create figure
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Sample points from original mesh
        samples = 10000
        orig_points = original_mesh.sample(samples)
        
        # Compute distances from original points to retopologized mesh
        distances = self._compute_point_to_mesh_distances(orig_points, retopologized_mesh)
        
        # Create a scatter plot with color based on distance
        scatter = ax.scatter(orig_points[:, 0], orig_points[:, 1], orig_points[:, 2],
                            c=distances, cmap='viridis', s=1)
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax, label='Distance')
        
        # Set title and labels
        ax.set_title("Distance Map")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Set equal aspect ratio
        ax.set_box_aspect([1, 1, 1])
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Distance map visualization saved to {output_path}")
    
    def visualize_feature_preservation(self, original_mesh: Union[trimesh.Trimesh, str], 
                                     retopologized_mesh: Union[trimesh.Trimesh, str], 
                                     output_path: Optional[Union[str, Path]] = None) -> None:
        """
        Visualize feature preservation between original and retopologized meshes.
        
        Args:
            original_mesh: Original mesh (trimesh object or path)
            retopologized_mesh: Retopologized mesh (trimesh object or path)
            output_path: Path to save the visualization (optional)
        """
        # Load meshes if paths are provided
        if isinstance(original_mesh, str):
            original_mesh = trimesh.load(original_mesh)
        if isinstance(retopologized_mesh, str):
            retopologized_mesh = trimesh.load(retopologized_mesh)
        
        # Create output directory if needed
        if output_path is None:
            output_path = self.output_dir / "feature_preservation.png"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 10))
        
        # Plot original mesh with sharp edges
        ax1 = fig.add_subplot(121, projection='3d')
        orig_edges = self._detect_sharp_edges(original_mesh)
        self._plot_mesh_with_edges(ax1, original_mesh, orig_edges, title="Original Mesh (Sharp Edges)")
        
        # Plot retopologized mesh with sharp edges
        ax2 = fig.add_subplot(122, projection='3d')
        retopo_edges = self._detect_sharp_edges(retopologized_mesh)
        self._plot_mesh_with_edges(ax2, retopologized_mesh, retopo_edges, title="Retopologized Mesh (Sharp Edges)")
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Feature preservation visualization saved to {output_path}")
    
    def visualize_mesh_stats(self, original_mesh: Union[trimesh.Trimesh, str], 
                           retopologized_mesh: Union[trimesh.Trimesh, str], 
                           output_path: Optional[Union[str, Path]] = None) -> None:
        """
        Visualize mesh statistics.
        
        Args:
            original_mesh: Original mesh (trimesh object or path)
            retopologized_mesh: Retopologized mesh (trimesh object or path)
            output_path: Path to save the visualization (optional)
        """
        # Load meshes if paths are provided
        if isinstance(original_mesh, str):
            original_mesh = trimesh.load(original_mesh)
        if isinstance(retopologized_mesh, str):
            retopologized_mesh = trimesh.load(retopologized_mesh)
        
        # Create output directory if needed
        if output_path is None:
            output_path = self.output_dir / "mesh_stats.png"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create figure
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        
        # Compute statistics
        orig_vertices = len(original_mesh.vertices)
        orig_faces = len(original_mesh.faces)
        retopo_vertices = len(retopologized_mesh.vertices)
        retopo_faces = len(retopologized_mesh.faces)
        
        # Create bar chart
        labels = ['Vertices', 'Faces']
        orig_values = [orig_vertices, orig_faces]
        retopo_values = [retopo_vertices, retopo_faces]
        
        x = np.arange(len(labels))
        width = 0.35
        
        ax.bar(x - width/2, orig_values, width, label='Original')
        ax.bar(x + width/2, retopo_values, width, label='Retopologized')
        
        # Add labels and title
        ax.set_ylabel('Count')
        ax.set_title('Mesh Statistics')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        
        # Add text annotations
        for i, v in enumerate(orig_values):
            ax.text(i - width/2, v, str(v), ha='center', va='bottom')
        
        for i, v in enumerate(retopo_values):
            ax.text(i + width/2, v, str(v), ha='center', va='bottom')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Mesh statistics visualization saved to {output_path}")
    
    def export_comparison(self, original_mesh: Union[trimesh.Trimesh, str], 
                         retopologized_mesh: Union[trimesh.Trimesh, str], 
                         output_dir: Optional[Union[str, Path]] = None) -> None:
        """
        Export a comprehensive comparison between original and retopologized meshes.
        
        Args:
            original_mesh: Original mesh (trimesh object or path)
            retopologized_mesh: Retopologized mesh (trimesh object or path)
            output_dir: Directory to save the visualizations (optional)
        """
        # Set output directory
        if output_dir is None:
            output_dir = self.output_dir
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export visualizations
        self.visualize_comparison(original_mesh, retopologized_mesh, 
                                output_path=output_dir / "comparison.png")
        self.visualize_distance_map(original_mesh, retopologized_mesh, 
                                  output_path=output_dir / "distance_map.png")
        self.visualize_feature_preservation(original_mesh, retopologized_mesh, 
                                          output_path=output_dir / "feature_preservation.png")
        self.visualize_mesh_stats(original_mesh, retopologized_mesh, 
                                output_path=output_dir / "mesh_stats.png")
        
        print(f"Comparison exported to {output_dir}") 