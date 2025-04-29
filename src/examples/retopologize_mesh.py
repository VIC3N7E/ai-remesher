#!/usr/bin/env python3
"""
Script to retopologize a mesh using a trained model.

This script loads a trained model checkpoint and uses it to retopologize a mesh
provided via command line arguments.
"""

import os
import argparse
import torch
import trimesh
from pathlib import Path

from src.models.retopology_model import RetopologyModel
from src.utils.feature_extraction import FeatureExtractor
from src.utils.mesh_export import MeshExporter
from src.utils.mesh_visualizer import MeshVisualizer


def load_model(checkpoint_path):
    """Load a trained model from a checkpoint."""
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    
    # Create model
    model = RetopologyModel()
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set to evaluation mode
    model.eval()
    
    return model


def retopologize_mesh(model, mesh_path, output_dir, visualize=True):
    """Retopologize a mesh using a trained model."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load mesh
    print(f"Loading mesh from {mesh_path}...")
    mesh = trimesh.load(mesh_path)
    
    # Extract features
    print("Extracting features...")
    feature_extractor = FeatureExtractor()
    features = feature_extractor.detect_features(mesh)
    
    # Prepare input for model
    # This would depend on your model's input requirements
    # For now, we'll just use the features directly
    with torch.no_grad():
        # Convert features to tensors
        # This is a placeholder - you'll need to adapt this to your model's input format
        input_tensors = {
            'vertices': torch.tensor(mesh.vertices, dtype=torch.float32),
            'faces': torch.tensor(mesh.faces, dtype=torch.long),
            'features': features
        }
        
        # Run model
        print("Running model...")
        output = model(input_tensors)
        
        # Process output
        # This is a placeholder - you'll need to adapt this to your model's output format
        new_vertices = output['vertices'].numpy()
        new_faces = output['faces'].numpy()
        
        # Create new mesh
        new_mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)
    
    # Export retopologized mesh
    print("Exporting retopologized mesh...")
    exporter = MeshExporter()
    output_path = os.path.join(output_dir, "retopologized_mesh.obj")
    exporter.export_mesh(new_mesh, output_path)
    
    # Visualize results
    if visualize:
        print("Visualizing results...")
        visualizer = MeshVisualizer()
        
        # Visualize original mesh
        visualizer.visualize_mesh(
            mesh=mesh,
            features=features,
            output_path=os.path.join(output_dir, "original_mesh.png"),
            show=True
        )
        
        # Visualize retopologized mesh
        visualizer.visualize_mesh(
            mesh=new_mesh,
            output_path=os.path.join(output_dir, "retopologized_mesh.png"),
            show=True
        )
    
    print(f"Retopologization complete. Results saved to {output_dir}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Retopologize a mesh using a trained model.")
    parser.add_argument("--mesh", type=str, required=True, help="Path to input mesh file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    parser.add_argument("--visualize", action="store_true", help="Whether to show visualizations")
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint)
    
    # Retopologize mesh
    retopologize_mesh(model, args.mesh, args.output, args.visualize)


if __name__ == "__main__":
    main() 