#!/usr/bin/env python3
"""
Utility module for visualizing meshes and mesh processing results.
"""

import os
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union, Dict, List, Optional, Tuple, Any
from mpl_toolkits.mplot3d import Axes3D
import pyvista as pv


class MeshVisualizer:
    """
    A class for visualizing meshes and mesh processing results.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the mesh visualizer.
        
        Args:
            config: Configuration dictionary with visualization parameters
        """
        self.config = config or {}
        self.supported_formats = ["png", "jpg", "pdf", "svg"]
    
    def visualize_mesh(
            self,
            mesh: trimesh.Trimesh,
            features: Optional[Dict[str, np.ndarray]] = None,
            output_path: Optional[str] = None,
            show: bool = True
        ) -> None:
        """Visualize a mesh with optional features.
        
        Args:
            mesh: The mesh to visualize
            features: Optional dictionary of features to visualize
            output_path: Optional path to save the visualization
            show: Whether to show the visualization
        """
        # Convert trimesh to pyvista
        vertices = np.array(mesh.vertices)
        faces = np.array(mesh.faces)
        
        # Format faces for PyVista (prepend number of vertices per face)
        faces_pv = np.column_stack((np.full(len(faces), 3), faces))
        faces_pv = faces_pv.flatten()
        
        # Create PyVista mesh
        grid = pv.PolyData(vertices, faces_pv)
        
        # Create plotter with off_screen mode if we're only saving
        off_screen = output_path is not None and not show
        plotter = pv.Plotter(off_screen=off_screen)
        
        # Add mesh to plotter
        plotter.add_mesh(grid, color='white', show_edges=True)
        
        # Add features if provided
        if features is not None:
            if 'sharp_edges' in features:
                edges = features['sharp_edges']
                if len(edges) > 0:
                    edge_points = vertices[edges].reshape(-1, 3)
                    edge_lines = np.column_stack((np.full(len(edge_points)//2, 2), 
                                                np.arange(len(edge_points)).reshape(-1, 2))).flatten()
                    edge_mesh = pv.PolyData(edge_points, edge_lines)
                    plotter.add_mesh(edge_mesh, color='red', line_width=2)
            
            if 'corners' in features:
                corners = features['corners']
                if len(corners) > 0:
                    corner_points = vertices[corners]
                    corner_mesh = pv.PolyData(corner_points)
                    plotter.add_mesh(corner_mesh, color='blue', point_size=10)
            
            if 'creases' in features:
                creases = features['creases']
                for crease in creases:
                    if len(crease) > 1:
                        crease_points = vertices[crease]
                        crease_lines = np.column_stack((np.full(len(crease)-1, 2),
                                                      np.column_stack((np.arange(len(crease)-1),
                                                                     np.arange(1, len(crease)))))).flatten()
                        crease_mesh = pv.PolyData(crease_points, crease_lines)
                        plotter.add_mesh(crease_mesh, color='green', line_width=2)
        
        # Save screenshot if output path is provided
        if output_path is not None:
            # Ensure the output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            if off_screen:
                plotter.show(auto_close=False)  # Render the scene
            plotter.screenshot(output_path)
        
        # Show the plot if requested
        if show:
            plotter.show()
        else:
            plotter.close()
    
    def visualize_features(self, 
                          mesh: trimesh.Trimesh, 
                          features: Dict[str, np.ndarray], 
                          output_path: Optional[Union[str, Path]] = None,
                          title: str = "Feature Visualization",
                          feature_colors: Optional[Dict[str, str]] = None,
                          feature_sizes: Optional[Dict[str, float]] = None,
                          feature_alphas: Optional[Dict[str, float]] = None,
                          camera_position: Optional[Tuple[float, float, float]] = None,
                          window_size: Tuple[int, int] = (800, 600),
                          format: str = "png") -> None:
        """
        Visualize mesh features.
        
        Args:
            mesh: Mesh to visualize
            features: Dictionary of feature arrays
            output_path: Path to save the visualization (if None, display only)
            title: Title for the visualization
            feature_colors: Dictionary of colors for each feature
            feature_sizes: Dictionary of sizes for each feature
            feature_alphas: Dictionary of alpha values for each feature
            camera_position: Camera position (x, y, z)
            window_size: Window size (width, height)
            format: Output format (png, jpg, pdf, svg)
        """
        # Create PyVista mesh
        vertices = mesh.vertices
        faces = np.hstack((np.full((len(mesh.faces), 1), 3), mesh.faces))
        pv_mesh = pv.PolyData(vertices, faces)
        
        # Create plotter
        plotter = pv.Plotter(window_size=window_size)
        plotter.add_title(title)
        
        # Set camera position if provided
        if camera_position is not None:
            plotter.camera_position = camera_position
        
        # Add mesh to plotter
        plotter.add_mesh(pv_mesh, color="lightgray", opacity=0.3)
        
        # Add features to plotter
        for feature_name, feature_array in features.items():
            # Get feature properties
            color = feature_colors.get(feature_name, "red") if feature_colors else "red"
            size = feature_sizes.get(feature_name, 5.0) if feature_sizes else 5.0
            alpha = feature_alphas.get(feature_name, 1.0) if feature_alphas else 1.0
            
            # Create feature points
            feature_points = pv.PolyData(feature_array)
            
            # Add feature to plotter
            plotter.add_mesh(feature_points, color=color, point_size=size, opacity=alpha)
        
        # Show or save visualization
        if output_path is not None:
            # Ensure format is lowercase
            format = format.lower()
            
            # Check if format is supported
            if format not in self.supported_formats:
                raise ValueError(f"Unsupported format: {format}. Supported formats: {self.supported_formats}")
            
            # Create output directory if needed
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save visualization
            plotter.screenshot(output_path)
        else:
            plotter.show()
    
    def visualize_comparison(self, 
                           original_mesh: trimesh.Trimesh, 
                           retopologized_mesh: trimesh.Trimesh, 
                           output_path: Optional[Union[str, Path]] = None,
                           title: str = "Mesh Comparison",
                           original_color: str = "lightblue",
                           retopologized_color: str = "lightgreen",
                           show_vertices: bool = True,
                           show_edges: bool = True,
                           show_faces: bool = True,
                           vertex_color: str = "red",
                           edge_color: str = "black",
                           vertex_size: float = 5.0,
                           edge_width: float = 1.0,
                           face_alpha: float = 0.5,
                           camera_position: Optional[Tuple[float, float, float]] = None,
                           window_size: Tuple[int, int] = (1200, 600),
                           format: str = "png") -> None:
        """
        Visualize original and retopologized meshes side by side.
        
        Args:
            original_mesh: Original mesh
            retopologized_mesh: Retopologized mesh
            output_path: Path to save the visualization (if None, display only)
            title: Title for the visualization
            original_color: Color for original mesh
            retopologized_color: Color for retopologized mesh
            show_vertices: Whether to show vertices
            show_edges: Whether to show edges
            show_faces: Whether to show faces
            vertex_color: Color for vertices
            edge_color: Color for edges
            vertex_size: Size of vertices
            edge_width: Width of edges
            face_alpha: Alpha value for faces
            camera_position: Camera position (x, y, z)
            window_size: Window size (width, height)
            format: Output format (png, jpg, pdf, svg)
        """
        # Create PyVista meshes
        original_vertices = original_mesh.vertices
        original_faces = np.hstack((np.full((len(original_mesh.faces), 1), 3), original_mesh.faces))
        original_pv_mesh = pv.PolyData(original_vertices, original_faces)
        
        retopologized_vertices = retopologized_mesh.vertices
        retopologized_faces = np.hstack((np.full((len(retopologized_mesh.faces), 1), 3), retopologized_mesh.faces))
        retopologized_pv_mesh = pv.PolyData(retopologized_vertices, retopologized_faces)
        
        # Create plotter
        plotter = pv.Plotter(shape=(1, 2), window_size=window_size)
        plotter.add_title(title)
        
        # Set camera position if provided
        if camera_position is not None:
            plotter.camera_position = camera_position
        
        # Add original mesh to left subplot
        plotter.subplot(0, 0)
        plotter.add_title("Original Mesh")
        
        if show_faces:
            plotter.add_mesh(original_pv_mesh, color=original_color, opacity=face_alpha)
        
        if show_edges:
            original_edges = original_pv_mesh.extract_feature_edges()
            plotter.add_mesh(original_edges, color=edge_color, line_width=edge_width)
        
        if show_vertices:
            original_vertices = pv.PolyData(original_vertices)
            plotter.add_mesh(original_vertices, color=vertex_color, point_size=vertex_size)
        
        # Add retopologized mesh to right subplot
        plotter.subplot(0, 1)
        plotter.add_title("Retopologized Mesh")
        
        if show_faces:
            plotter.add_mesh(retopologized_pv_mesh, color=retopologized_color, opacity=face_alpha)
        
        if show_edges:
            retopologized_edges = retopologized_pv_mesh.extract_feature_edges()
            plotter.add_mesh(retopologized_edges, color=edge_color, line_width=edge_width)
        
        if show_vertices:
            retopologized_vertices = pv.PolyData(retopologized_vertices)
            plotter.add_mesh(retopologized_vertices, color=vertex_color, point_size=vertex_size)
        
        # Show or save visualization
        if output_path is not None:
            # Ensure format is lowercase
            format = format.lower()
            
            # Check if format is supported
            if format not in self.supported_formats:
                raise ValueError(f"Unsupported format: {format}. Supported formats: {self.supported_formats}")
            
            # Create output directory if needed
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save visualization
            plotter.screenshot(output_path)
        else:
            plotter.show()
    
    def visualize_curvature(self, 
                           mesh: trimesh.Trimesh, 
                           curvature: np.ndarray, 
                           output_path: Optional[Union[str, Path]] = None,
                           title: str = "Curvature Visualization",
                           colormap: str = "jet",
                           camera_position: Optional[Tuple[float, float, float]] = None,
                           window_size: Tuple[int, int] = (800, 600),
                           format: str = "png") -> None:
        """
        Visualize mesh curvature.
        
        Args:
            mesh: Mesh to visualize
            curvature: Curvature values for each vertex
            output_path: Path to save the visualization (if None, display only)
            title: Title for the visualization
            colormap: Colormap for curvature values
            camera_position: Camera position (x, y, z)
            window_size: Window size (width, height)
            format: Output format (png, jpg, pdf, svg)
        """
        # Create PyVista mesh
        vertices = mesh.vertices
        faces = np.hstack((np.full((len(mesh.faces), 1), 3), mesh.faces))
        pv_mesh = pv.PolyData(vertices, faces)
        
        # Add curvature values to mesh
        pv_mesh.point_data["curvature"] = curvature
        
        # Create plotter
        plotter = pv.Plotter(window_size=window_size)
        plotter.add_title(title)
        
        # Set camera position if provided
        if camera_position is not None:
            plotter.camera_position = camera_position
        
        # Add mesh to plotter with curvature coloring
        plotter.add_mesh(pv_mesh, scalars="curvature", cmap=colormap)
        
        # Add colorbar
        plotter.add_scalar_bar(title="Curvature", vertical=True)
        
        # Show or save visualization
        if output_path is not None:
            # Ensure format is lowercase
            format = format.lower()
            
            # Check if format is supported
            if format not in self.supported_formats:
                raise ValueError(f"Unsupported format: {format}. Supported formats: {self.supported_formats}")
            
            # Create output directory if needed
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save visualization
            plotter.screenshot(output_path)
        else:
            plotter.show()
    
    def visualize_feature_edges(self, 
                              mesh: trimesh.Trimesh, 
                              feature_edges: np.ndarray, 
                              output_path: Optional[Union[str, Path]] = None,
                              title: str = "Feature Edge Visualization",
                              feature_color: str = "red",
                              non_feature_color: str = "black",
                              feature_width: float = 2.0,
                              non_feature_width: float = 1.0,
                              camera_position: Optional[Tuple[float, float, float]] = None,
                              window_size: Tuple[int, int] = (800, 600),
                              format: str = "png") -> None:
        """
        Visualize feature edges.
        
        Args:
            mesh: Mesh to visualize
            feature_edges: Boolean array indicating feature edges
            output_path: Path to save the visualization (if None, display only)
            title: Title for the visualization
            feature_color: Color for feature edges
            non_feature_color: Color for non-feature edges
            feature_width: Width for feature edges
            non_feature_width: Width for non-feature edges
            camera_position: Camera position (x, y, z)
            window_size: Window size (width, height)
            format: Output format (png, jpg, pdf, svg)
        """
        # Create PyVista mesh
        vertices = mesh.vertices
        faces = np.hstack((np.full((len(mesh.faces), 1), 3), mesh.faces))
        pv_mesh = pv.PolyData(vertices, faces)
        
        # Extract edges
        edges = pv_mesh.extract_feature_edges()
        
        # Create plotter
        plotter = pv.Plotter(window_size=window_size)
        plotter.add_title(title)
        
        # Set camera position if provided
        if camera_position is not None:
            plotter.camera_position = camera_position
        
        # Add mesh to plotter
        plotter.add_mesh(pv_mesh, color="lightgray", opacity=0.3)
        
        # Add feature edges to plotter
        feature_edge_indices = np.where(feature_edges)[0]
        non_feature_edge_indices = np.where(~feature_edges)[0]
        
        if len(feature_edge_indices) > 0:
            feature_edge_mesh = edges.extract_points(feature_edge_indices)
            plotter.add_mesh(feature_edge_mesh, color=feature_color, line_width=feature_width)
        
        if len(non_feature_edge_indices) > 0:
            non_feature_edge_mesh = edges.extract_points(non_feature_edge_indices)
            plotter.add_mesh(non_feature_edge_mesh, color=non_feature_color, line_width=non_feature_width)
        
        # Show or save visualization
        if output_path is not None:
            # Ensure format is lowercase
            format = format.lower()
            
            # Check if format is supported
            if format not in self.supported_formats:
                raise ValueError(f"Unsupported format: {format}. Supported formats: {self.supported_formats}")
            
            # Create output directory if needed
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save visualization
            plotter.screenshot(output_path)
        else:
            plotter.show()
    
    def visualize_feature_corners(self, 
                                mesh: trimesh.Trimesh, 
                                feature_corners: np.ndarray, 
                                output_path: Optional[Union[str, Path]] = None,
                                title: str = "Feature Corner Visualization",
                                corner_color: str = "red",
                                corner_size: float = 10.0,
                                camera_position: Optional[Tuple[float, float, float]] = None,
                                window_size: Tuple[int, int] = (800, 600),
                                format: str = "png") -> None:
        """
        Visualize feature corners.
        
        Args:
            mesh: Mesh to visualize
            feature_corners: Boolean array indicating feature corners
            output_path: Path to save the visualization (if None, display only)
            title: Title for the visualization
            corner_color: Color for feature corners
            corner_size: Size for feature corners
            camera_position: Camera position (x, y, z)
            window_size: Window size (width, height)
            format: Output format (png, jpg, pdf, svg)
        """
        # Create PyVista mesh
        vertices = mesh.vertices
        faces = np.hstack((np.full((len(mesh.faces), 1), 3), mesh.faces))
        pv_mesh = pv.PolyData(vertices, faces)
        
        # Create plotter
        plotter = pv.Plotter(window_size=window_size)
        plotter.add_title(title)
        
        # Set camera position if provided
        if camera_position is not None:
            plotter.camera_position = camera_position
        
        # Add mesh to plotter
        plotter.add_mesh(pv_mesh, color="lightgray", opacity=0.3)
        
        # Add feature corners to plotter
        corner_indices = np.where(feature_corners)[0]
        
        if len(corner_indices) > 0:
            corner_points = pv.PolyData(vertices[corner_indices])
            plotter.add_mesh(corner_points, color=corner_color, point_size=corner_size)
        
        # Show or save visualization
        if output_path is not None:
            # Ensure format is lowercase
            format = format.lower()
            
            # Check if format is supported
            if format not in self.supported_formats:
                raise ValueError(f"Unsupported format: {format}. Supported formats: {self.supported_formats}")
            
            # Create output directory if needed
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save visualization
            plotter.screenshot(output_path)
        else:
            plotter.show()
    
    def visualize_feature_creases(self, 
                                mesh: trimesh.Trimesh, 
                                feature_creases: np.ndarray, 
                                output_path: Optional[Union[str, Path]] = None,
                                title: str = "Feature Crease Visualization",
                                crease_color: str = "blue",
                                crease_width: float = 2.0,
                                camera_position: Optional[Tuple[float, float, float]] = None,
                                window_size: Tuple[int, int] = (800, 600),
                                format: str = "png") -> None:
        """
        Visualize feature creases.
        
        Args:
            mesh: Mesh to visualize
            feature_creases: Boolean array indicating feature creases
            output_path: Path to save the visualization (if None, display only)
            title: Title for the visualization
            crease_color: Color for feature creases
            crease_width: Width for feature creases
            camera_position: Camera position (x, y, z)
            window_size: Window size (width, height)
            format: Output format (png, jpg, pdf, svg)
        """
        # Create PyVista mesh
        vertices = mesh.vertices
        faces = np.hstack((np.full((len(mesh.faces), 1), 3), mesh.faces))
        pv_mesh = pv.PolyData(vertices, faces)
        
        # Create plotter
        plotter = pv.Plotter(window_size=window_size)
        plotter.add_title(title)
        
        # Set camera position if provided
        if camera_position is not None:
            plotter.camera_position = camera_position
        
        # Add mesh to plotter
        plotter.add_mesh(pv_mesh, color="lightgray", opacity=0.3)
        
        # Add feature creases to plotter
        crease_indices = np.where(feature_creases)[0]
        
        if len(crease_indices) > 0:
            crease_edges = pv_mesh.extract_feature_edges()
            crease_edges = crease_edges.extract_points(crease_indices)
            plotter.add_mesh(crease_edges, color=crease_color, line_width=crease_width)
        
        # Show or save visualization
        if output_path is not None:
            # Ensure format is lowercase
            format = format.lower()
            
            # Check if format is supported
            if format not in self.supported_formats:
                raise ValueError(f"Unsupported format: {format}. Supported formats: {self.supported_formats}")
            
            # Create output directory if needed
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save visualization
            plotter.screenshot(output_path)
        else:
            plotter.show() 