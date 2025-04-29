#!/usr/bin/env python3
"""
Dataset generation utilities.

This module provides functions for generating synthetic datasets for training and testing.
"""

import numpy as np
import trimesh
import pyvista as pv
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


def create_primitive_mesh(
    primitive_type: str,
    size: float = 1.0,
    resolution: int = 32
) -> trimesh.Trimesh:
    """Create a primitive mesh.
    
    Args:
        primitive_type: Type of primitive ('sphere', 'cube', 'cylinder', 'torus')
        size: Size of the primitive
        resolution: Resolution for curved primitives
        
    Returns:
        Trimesh object
    """
    if primitive_type == 'sphere':
        # Create sphere using parametric equations
        phi = np.linspace(0, 2*np.pi, resolution)
        theta = np.linspace(0, np.pi, resolution)
        phi, theta = np.meshgrid(phi, theta)
        
        x = size * np.sin(theta) * np.cos(phi)
        y = size * np.sin(theta) * np.sin(phi)
        z = size * np.cos(theta)
        
        # Create structured grid
        grid = pv.StructuredGrid(x, y, z)
        mesh = grid.extract_surface()
        
        # Convert to trimesh
        vertices = np.array(mesh.points)
        faces = np.array(mesh.faces.reshape(-1, 4))[:, 1:4]
        return trimesh.Trimesh(vertices=vertices, faces=faces)
        
    elif primitive_type == 'cube':
        return trimesh.creation.box(extents=size)
        
    elif primitive_type == 'cylinder':
        return trimesh.creation.cylinder(radius=size/2, height=size)
        
    elif primitive_type == 'torus':
        # Create torus using parametric equations
        u = np.linspace(0, 2*np.pi, resolution)
        v = np.linspace(0, 2*np.pi, resolution)
        u, v = np.meshgrid(u, v)
        
        R = size/2  # major radius
        r = size/4  # minor radius
        
        x = (R + r*np.cos(v)) * np.cos(u)
        y = (R + r*np.cos(v)) * np.sin(u)
        z = r * np.sin(v)
        
        # Create structured grid
        grid = pv.StructuredGrid(x, y, z)
        mesh = grid.extract_surface()
        
        # Convert to trimesh
        vertices = np.array(mesh.points)
        faces = np.array(mesh.faces.reshape(-1, 4))[:, 1:4]
        return trimesh.Trimesh(vertices=vertices, faces=faces)
        
    else:
        raise ValueError(f"Unknown primitive type: {primitive_type}")


def create_complex_mesh(
    primitive_types: List[str],
    positions: List[Tuple[float, float, float]],
    sizes: List[float],
    operations: List[str] = None
) -> trimesh.Trimesh:
    """Create a complex mesh by combining primitives.
    
    Args:
        primitive_types: List of primitive types
        positions: List of (x, y, z) positions
        sizes: List of sizes
        operations: List of boolean operations ('union', 'difference', 'intersection')
        
    Returns:
        Trimesh object
    """
    if len(primitive_types) != len(positions) or len(primitive_types) != len(sizes):
        raise ValueError("Lists must have the same length")
        
    if operations is None:
        operations = ['union'] * (len(primitive_types) - 1)
    elif len(operations) != len(primitive_types) - 1:
        raise ValueError("Number of operations must be one less than number of primitives")
    
    # Create first primitive
    mesh = create_primitive_mesh(primitive_types[0], sizes[0])
    mesh.apply_translation(positions[0])
    
    # Combine with remaining primitives
    for i in range(1, len(primitive_types)):
        next_mesh = create_primitive_mesh(primitive_types[i], sizes[i])
        next_mesh.apply_translation(positions[i])
        
        if operations[i-1] == 'union':
            mesh = mesh.union(next_mesh)
        elif operations[i-1] == 'difference':
            mesh = mesh.difference(next_mesh)
        elif operations[i-1] == 'intersection':
            mesh = mesh.intersection(next_mesh)
        else:
            raise ValueError(f"Unknown operation: {operations[i-1]}")
    
    return mesh


def generate_dataset(
    output_dir: Union[str, Path],
    num_samples: int = 100,
    primitive_types: Optional[List[str]] = None,
    size_range: Tuple[float, float] = (0.5, 2.0),
    position_range: Tuple[float, float] = (-2.0, 2.0)
) -> Dict[str, List[Path]]:
    """Generate a dataset of synthetic meshes.
    
    Args:
        output_dir: Directory to save the dataset
        num_samples: Number of samples to generate
        primitive_types: List of primitive types to use
        size_range: Range of sizes for primitives
        position_range: Range of positions for primitives
        
    Returns:
        Dictionary mapping primitive types to lists of mesh paths
    """
    if primitive_types is None:
        primitive_types = ['sphere', 'cube', 'cylinder', 'torus']
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    mesh_paths = {pt: [] for pt in primitive_types}
    
    for i in range(num_samples):
        # Randomly select primitive type
        primitive_type = np.random.choice(primitive_types)
        
        # Create mesh
        size = np.random.uniform(*size_range)
        mesh = create_primitive_mesh(primitive_type, size)
        
        # Save mesh
        mesh_path = output_dir / f"{primitive_type}_{i:04d}.obj"
        mesh.export(mesh_path)
        mesh_paths[primitive_type].append(mesh_path)
    
    return mesh_paths


def generate_complex_dataset(
    output_dir: Union[str, Path],
    num_samples: int = 50,
    primitive_types: Optional[List[str]] = None,
    num_primitives_range: Tuple[int, int] = (2, 5),
    size_range: Tuple[float, float] = (0.5, 2.0),
    position_range: Tuple[float, float] = (-2.0, 2.0)
) -> List[Path]:
    """Generate a dataset of complex synthetic meshes.
    
    Args:
        output_dir: Directory to save the dataset
        num_samples: Number of samples to generate
        primitive_types: List of primitive types to use
        num_primitives_range: Range of number of primitives per mesh
        size_range: Range of sizes for primitives
        position_range: Range of positions for primitives
        
    Returns:
        List of mesh paths
    """
    if primitive_types is None:
        primitive_types = ['sphere', 'cube', 'cylinder', 'torus']
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    mesh_paths = []
    
    for i in range(num_samples):
        # Randomly select number of primitives
        num_primitives = np.random.randint(*num_primitives_range)
        
        # Randomly select primitive types
        selected_types = np.random.choice(primitive_types, num_primitives)
        
        # Generate random positions and sizes
        positions = [tuple(np.random.uniform(*position_range, 3)) for _ in range(num_primitives)]
        sizes = [np.random.uniform(*size_range) for _ in range(num_primitives)]
        
        # Create mesh
        mesh = create_complex_mesh(selected_types, positions, sizes)
        
        # Save mesh
        mesh_path = output_dir / f"complex_{i:04d}.obj"
        mesh.export(mesh_path)
        mesh_paths.append(mesh_path)
    
    return mesh_paths 