#!/usr/bin/env python3
"""
Feature visualization utilities.

This module provides functions for visualizing mesh features using matplotlib and PyVista.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import trimesh
import pyvista as pv
from pathlib import Path
from typing import Dict, Any, Optional, Union


def visualize_features(
    mesh: Union[str, trimesh.Trimesh],
    features: Dict[str, Any],
    output_dir: Optional[Union[str, Path]] = None
) -> None:
    """Visualize extracted features on the mesh.
    
    Args:
        mesh: Path to the mesh file or trimesh.Trimesh object
        features: Dictionary of extracted features
        output_dir: Directory to save visualization images
    """
    # Load the mesh if path is provided
    if isinstance(mesh, str):
        mesh = trimesh.load(str(mesh))
    
    # Create a PyVista mesh for visualization
    vertices = np.array(mesh.vertices)
    faces = np.array(mesh.faces)
    faces_padded = np.column_stack((np.full(len(faces), 3), faces))
    pv_mesh = pv.PolyData(vertices, faces_padded.flatten())
    
    # Create output directory if specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Visualize vertex features
    if 'vertex_features' in features:
        vertex_features = features['vertex_features']
        
        # Plot vertex positions
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c='b', marker='.')
        ax.set_title('Vertex Positions')
        if output_dir:
            plt.savefig(output_dir / 'vertex_positions.png')
        plt.close()
        
        # Plot vertex normals if available
        if 'normals' in vertex_features:
            normals = vertex_features['normals']
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
            ax.quiver(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                     normals[:, 0], normals[:, 1], normals[:, 2],
                     length=0.1, color='r')
            ax.set_title('Vertex Normals')
            if output_dir:
                plt.savefig(output_dir / 'vertex_normals.png')
            plt.close()
    
    # Visualize face features
    if 'face_features' in features:
        face_features = features['face_features']
        
        # Plot face normals if available
        if 'normals' in face_features:
            normals = face_features['normals']
            centroids = face_features['centroids']
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
            ax.quiver(centroids[:, 0], centroids[:, 1], centroids[:, 2],
                     normals[:, 0], normals[:, 1], normals[:, 2],
                     length=0.1, color='g')
            ax.set_title('Face Normals')
            if output_dir:
                plt.savefig(output_dir / 'face_normals.png')
            plt.close()
    
    # Visualize curvatures
    if 'curvatures' in features:
        curvatures = features['curvatures']
        
        # Plot mean curvature
        if 'mean_curvature' in curvatures:
            mean_curvature = curvatures['mean_curvature']
            pv_mesh['mean_curvature'] = mean_curvature
            plotter = pv.Plotter(off_screen=True)
            plotter.add_mesh(pv_mesh, scalars='mean_curvature', cmap='viridis')
            plotter.add_title('Mean Curvature')
            if output_dir:
                plotter.screenshot(str(output_dir / 'mean_curvature.png'))
            plotter.close()
        
        # Plot Gaussian curvature
        if 'gaussian_curvature' in curvatures:
            gaussian_curvature = curvatures['gaussian_curvature']
            pv_mesh['gaussian_curvature'] = gaussian_curvature
            plotter = pv.Plotter(off_screen=True)
            plotter.add_mesh(pv_mesh, scalars='gaussian_curvature', cmap='viridis')
            plotter.add_title('Gaussian Curvature')
            if output_dir:
                plotter.screenshot(str(output_dir / 'gaussian_curvature.png'))
            plotter.close()
    
    # Visualize detected features
    if 'detected_features' in features:
        detected_features = features['detected_features']
        
        # Plot sharp edges
        if 'sharp_edges' in detected_features:
            sharp_edges = detected_features['sharp_edges']
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
            for edge in sharp_edges:
                ax.plot([vertices[edge[0], 0], vertices[edge[1], 0]],
                       [vertices[edge[0], 1], vertices[edge[1], 1]],
                       [vertices[edge[0], 2], vertices[edge[1], 2]], 'r-', linewidth=2)
            ax.set_title('Sharp Edges')
            if output_dir:
                plt.savefig(output_dir / 'sharp_edges.png')
            plt.close()
        
        # Plot corners
        if 'corners' in detected_features:
            corners = detected_features['corners']
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(vertices[corners, 0], vertices[corners, 1], vertices[corners, 2],
                      c='r', marker='o', s=100)
            ax.set_title('Corners')
            if output_dir:
                plt.savefig(output_dir / 'corners.png')
            plt.close() 