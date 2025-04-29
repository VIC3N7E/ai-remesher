#!/usr/bin/env python3
"""
Test script to demonstrate the core functionality of the mesh processing pipeline.
"""

import os
import yaml
import trimesh
import numpy as np
from pathlib import Path
from typing import Dict, Optional

# Use absolute imports instead of relative imports
from src.utils.feature_extraction import FeatureExtractor
from src.utils.mesh_visualizer import MeshVisualizer
from src.utils.mesh_export import MeshExporter


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def test_mesh_processing(
    mesh_path: str,
    output_dir: str = "output",
    visualize: bool = True,
    save_visualization: bool = True
) -> None:
    """Test the mesh processing pipeline.
    
    Args:
        mesh_path: Path to the input mesh
        output_dir: Directory to save outputs
        visualize: Whether to show the visualization
        save_visualization: Whether to save the visualization
    """
    print(f"Loading mesh from {mesh_path}...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load mesh
    mesh = trimesh.load_mesh(mesh_path)
    
    # Extract features
    print("Extracting features...")
    feature_extractor = FeatureExtractor()
    features = feature_extractor.detect_features(mesh)
    
    # Visualize results
    if visualize or save_visualization:
        print("Visualizing mesh...")
        visualizer = MeshVisualizer()
        
        # Visualize mesh with features
        output_path = os.path.join(output_dir, "mesh_with_features.png") if save_visualization else None
        visualizer.visualize_mesh(
            mesh=mesh,
            features=features,
            output_path=output_path,
            show=visualize
        )
        
        # Visualize features separately
        if save_visualization:
            # Visualize sharp edges
            output_path = os.path.join(output_dir, "sharp_edges.png")
            visualizer.visualize_mesh(
                mesh=mesh,
                features={'sharp_edges': features['sharp_edges']},
                output_path=output_path,
                show=False
            )
            
            # Visualize corners
            output_path = os.path.join(output_dir, "corners.png")
            visualizer.visualize_mesh(
                mesh=mesh,
                features={'corners': features['corners']},
                output_path=output_path,
                show=False
            )
            
            # Visualize creases
            output_path = os.path.join(output_dir, "creases.png")
            visualizer.visualize_mesh(
                mesh=mesh,
                features={'creases': features['creases']},
                output_path=output_path,
                show=False
            )
    
    # Export mesh with features
    print("Exporting mesh with features...")
    exporter = MeshExporter()
    exporter.export_with_features(
        mesh=mesh,
        features=features,
        output_path=os.path.join(output_dir, "mesh_with_features.obj")
    )
    
    print("Done!")


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path
    
    # Add the project root directory to the Python path
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
    
    parser = argparse.ArgumentParser(description="Test the mesh processing pipeline.")
    parser.add_argument("--mesh", type=str, required=True, help="Path to input mesh file")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    parser.add_argument("--visualize", action="store_true", help="Whether to show the visualization")
    parser.add_argument("--save_visualization", action="store_true", help="Whether to save the visualization")
    
    args = parser.parse_args()
    
    test_mesh_processing(
        mesh_path=args.mesh,
        output_dir=args.output,
        visualize=args.visualize,
        save_visualization=args.save_visualization
    ) 