#!/usr/bin/env python3
"""
Utility module for exporting meshes in various formats with different options.
"""

import os
import numpy as np
import trimesh
from pathlib import Path
from typing import Union, Dict, List, Optional, Tuple, Any


class MeshExporter:
    """
    A class for exporting meshes in various formats with different options.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the mesh exporter.
        
        Args:
            config: Configuration dictionary with export parameters
        """
        self.config = config or {}
        self.supported_formats = ["obj", "ply", "stl"]
    
    def export_mesh(self, 
                   mesh: trimesh.Trimesh, 
                   output_path: Union[str, Path], 
                   format: str = "obj",
                   include_normals: bool = True,
                   include_textures: bool = True,
                   include_vertex_colors: bool = True,
                   include_face_colors: bool = True,
                   include_metadata: bool = True,
                   precision: int = 6) -> None:
        """
        Export a mesh to a file with specified options.
        
        Args:
            mesh: Mesh to export
            output_path: Path to save the mesh
            format: Output format (obj, ply, stl)
            include_normals: Whether to include vertex normals
            include_textures: Whether to include texture coordinates
            include_vertex_colors: Whether to include vertex colors
            include_face_colors: Whether to include face colors
            include_metadata: Whether to include metadata
            precision: Number of decimal places for vertex coordinates
        """
        # Create output directory if needed
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Ensure format is lowercase
        format = format.lower()
        
        # Check if format is supported
        if format not in self.supported_formats:
            raise ValueError(f"Unsupported format: {format}. Supported formats: {self.supported_formats}")
        
        # Prepare export options
        export_options = {
            "include_normals": include_normals,
            "include_textures": include_textures,
            "include_vertex_colors": include_vertex_colors,
            "include_face_colors": include_face_colors,
            "include_metadata": include_metadata,
            "precision": precision
        }
        
        # Export mesh based on format
        if format == "obj":
            self._export_obj(mesh, output_path, export_options)
        elif format == "ply":
            self._export_ply(mesh, output_path, export_options)
        elif format == "stl":
            self._export_stl(mesh, output_path, export_options)
    
    def _export_obj(self, mesh: trimesh.Trimesh, output_path: Path, options: Dict) -> None:
        """
        Export a mesh to OBJ format.
        
        Args:
            mesh: Mesh to export
            output_path: Path to save the mesh
            options: Export options
        """
        # Create a copy of the mesh to avoid modifying the original
        export_mesh = mesh.copy()
        
        # Ensure mesh has vertex normals if requested
        if options["include_normals"] and not hasattr(export_mesh, "vertex_normals"):
            export_mesh.vertex_normals = export_mesh.face_normals.mean(axis=0)
        
        # Export mesh
        with open(output_path, "w") as f:
            # Write header
            f.write("# Exported by AI Remesher\n")
            
            # Write vertices
            for vertex in export_mesh.vertices:
                f.write(f"v {vertex[0]:.{options['precision']}f} {vertex[1]:.{options['precision']}f} {vertex[2]:.{options['precision']}f}\n")
            
            # Write vertex normals if requested
            if options["include_normals"] and hasattr(export_mesh, "vertex_normals"):
                for normal in export_mesh.vertex_normals:
                    f.write(f"vn {normal[0]:.{options['precision']}f} {normal[1]:.{options['precision']}f} {normal[2]:.{options['precision']}f}\n")
            
            # Write texture coordinates if requested and available
            if options["include_textures"] and hasattr(export_mesh, "visual") and hasattr(export_mesh.visual, "uv"):
                for uv in export_mesh.visual.uv:
                    f.write(f"vt {uv[0]:.{options['precision']}f} {uv[1]:.{options['precision']}f}\n")
            
            # Write faces
            for i, face in enumerate(export_mesh.faces):
                # OBJ indices are 1-based
                v1, v2, v3 = face + 1
                
                # Write face with appropriate indices
                if options["include_normals"] and hasattr(export_mesh, "vertex_normals"):
                    if options["include_textures"] and hasattr(export_mesh, "visual") and hasattr(export_mesh.visual, "uv"):
                        f.write(f"f {v1}/{v1}/{v1} {v2}/{v2}/{v2} {v3}/{v3}/{v3}\n")
                    else:
                        f.write(f"f {v1}//{v1} {v2}//{v2} {v3}//{v3}\n")
                else:
                    f.write(f"f {v1} {v2} {v3}\n")
    
    def _export_ply(self, mesh: trimesh.Trimesh, output_path: Path, options: Dict) -> None:
        """
        Export a mesh to PLY format.
        
        Args:
            mesh: Mesh to export
            output_path: Path to save the mesh
            options: Export options
        """
        # Create a copy of the mesh to avoid modifying the original
        export_mesh = mesh.copy()
        
        # Ensure mesh has vertex normals if requested
        if options["include_normals"] and not hasattr(export_mesh, "vertex_normals"):
            export_mesh.vertex_normals = export_mesh.face_normals.mean(axis=0)
        
        # Export mesh using trimesh's built-in PLY exporter
        export_mesh.export(output_path)
    
    def _export_stl(self, mesh: trimesh.Trimesh, output_path: Path, options: Dict) -> None:
        """
        Export a mesh to STL format.
        
        Args:
            mesh: Mesh to export
            output_path: Path to save the mesh
            options: Export options
        """
        # Create a copy of the mesh to avoid modifying the original
        export_mesh = mesh.copy()
        
        # Export mesh using trimesh's built-in STL exporter
        export_mesh.export(output_path)
    
    def export_batch(self, 
                    meshes: List[trimesh.Trimesh], 
                    output_dir: Union[str, Path], 
                    format: str = "obj",
                    prefix: str = "",
                    suffix: str = "",
                    include_normals: bool = True,
                    include_textures: bool = True,
                    include_vertex_colors: bool = True,
                    include_face_colors: bool = True,
                    include_metadata: bool = True,
                    precision: int = 6) -> List[Path]:
        """
        Export a batch of meshes to files.
        
        Args:
            meshes: List of meshes to export
            output_dir: Directory to save the meshes
            format: Output format (obj, ply, stl)
            prefix: Prefix for output filenames
            suffix: Suffix for output filenames
            include_normals: Whether to include vertex normals
            include_textures: Whether to include texture coordinates
            include_vertex_colors: Whether to include vertex colors
            include_face_colors: Whether to include face colors
            include_metadata: Whether to include metadata
            precision: Number of decimal places for vertex coordinates
            
        Returns:
            List of paths to exported meshes
        """
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export each mesh
        exported_paths = []
        for i, mesh in enumerate(meshes):
            # Create output path
            output_path = output_dir / f"{prefix}mesh_{i}{suffix}.{format}"
            
            # Export mesh
            self.export_mesh(
                mesh, 
                output_path, 
                format, 
                include_normals, 
                include_textures, 
                include_vertex_colors, 
                include_face_colors, 
                include_metadata, 
                precision
            )
            
            exported_paths.append(output_path)
        
        return exported_paths
    
    def export_with_features(self, 
                           mesh: trimesh.Trimesh, 
                           features: Dict[str, Any], 
                           output_path: Union[str, Path], 
                           format: str = "obj") -> None:
        """
        Export a mesh with feature information.
        
        Args:
            mesh: Mesh to export
            features: Feature information to include in the export
            output_path: Path to save the mesh
            format: Output format (obj, ply, stl)
        """
        # Create a copy of the mesh to avoid modifying the original
        export_mesh = mesh.copy()
        
        # Add feature information to mesh metadata
        if not hasattr(export_mesh, "metadata"):
            export_mesh.metadata = {}
        
        export_mesh.metadata["features"] = features
        
        # Export mesh
        self.export_mesh(export_mesh, output_path, format)
    
    def export_comparison(self, 
                         original_mesh: trimesh.Trimesh, 
                         retopologized_mesh: trimesh.Trimesh, 
                         output_dir: Union[str, Path], 
                         format: str = "obj") -> Tuple[Path, Path]:
        """
        Export original and retopologized meshes for comparison.
        
        Args:
            original_mesh: Original mesh
            retopologized_mesh: Retopologized mesh
            output_dir: Directory to save the meshes
            format: Output format (obj, ply, stl)
            
        Returns:
            Tuple of paths to exported meshes
        """
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export original mesh
        original_path = output_dir / f"original.{format}"
        self.export_mesh(original_mesh, original_path, format)
        
        # Export retopologized mesh
        retopologized_path = output_dir / f"retopologized.{format}"
        self.export_mesh(retopologized_mesh, retopologized_path, format)
        
        return original_path, retopologized_path 