#!/usr/bin/env python3
"""
Script to create sample meshes for testing.

This module provides functionality to create various geometric primitives
as sample meshes for testing the AI Remesher pipeline.
"""

import os
import numpy as np
import pyvista as pv
from pathlib import Path


def create_sample_meshes(output_dir: str = "sample_meshes") -> dict:
    """
    Create sample meshes for testing.
    
    Args:
        output_dir: Directory to save the sample meshes
        
    Returns:
        dict: Dictionary mapping mesh names to their file paths
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create meshes
    meshes = {
        'sphere': pv.Sphere(radius=1.0, center=(0, 0, 0), phi_resolution=50, theta_resolution=50),
        'cube': pv.Cube(center=(0, 0, 0), x_length=1.0, y_length=1.0, z_length=1.0),
        'cylinder': pv.Cylinder(center=(0, 0, 0), direction=(0, 0, 1), radius=0.5, height=2.0, resolution=50),
        'ring': pv.CircularArc(
            pointa=(1, 0, 0),
            pointb=(0, 1, 0),
            center=(0, 0, 0),
            resolution=50
        ).tube(radius=0.3, n_sides=50)
    }
    
    # Save meshes and collect paths
    mesh_paths = {}
    for name, mesh in meshes.items():
        filepath = output_dir / f"{name}.obj"
        mesh.save(filepath)
        mesh_paths[name] = str(filepath)
    
    print(f"Sample meshes saved to {output_dir}")
    return mesh_paths


if __name__ == "__main__":
    create_sample_meshes() 