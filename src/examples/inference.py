#!/usr/bin/env python3
"""
Inference script to apply the trained model to new meshes and export the retopologized results.
"""

import argparse
import os
import yaml
import torch
import numpy as np
import trimesh
from pathlib import Path
from typing import Union, List, Dict, Optional, Tuple
from tqdm import tqdm

from src.models.mesh_network import MeshNetwork
from src.data.mesh_dataset import MeshDataset
from src.utils.feature_extraction import FeatureExtractor
from src.utils.visualization import MeshVisualizer
from src.utils.evaluation import MeshEvaluator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Apply trained model to new meshes for retopology.")
    parser.add_argument("--input", type=str, required=True, help="Path to input mesh file or directory")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model checkpoint")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to the configuration file")
    parser.add_argument("--output", type=str, default="output/inference", help="Path to the output directory")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for inference (cuda, cpu)")
    parser.add_argument("--format", type=str, default="obj", help="Output format (obj, ply)")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate retopology quality")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load the configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_model(model_path: str, config: Dict, device: str) -> torch.nn.Module:
    """
    Load the trained model from a checkpoint.
    
    Args:
        model_path: Path to the model checkpoint
        config: Configuration dictionary
        device: Device to load the model on
        
    Returns:
        Loaded model
    """
    # Create model
    model = MeshNetwork(
        input_channels=config["model"]["input_channels"],
        hidden_channels=config["model"]["hidden_channels"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"],
        activation=config["model"]["activation"],
        normalization=config["model"]["normalization"]
    )
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Move model to device
    model = model.to(device)
    model.eval()
    
    return model


def process_mesh(mesh_path: Union[str, Path], 
                model: torch.nn.Module, 
                feature_extractor: FeatureExtractor,
                config: Dict,
                device: str) -> Tuple[trimesh.Trimesh, Dict]:
    """
    Process a single mesh with the trained model.
    
    Args:
        mesh_path: Path to the input mesh
        model: Trained model
        feature_extractor: Feature extractor
        config: Configuration dictionary
        device: Device to use for inference
        
    Returns:
        Retopologized mesh and feature information
    """
    # Load mesh
    mesh = trimesh.load(mesh_path)
    
    # Extract features
    features = feature_extractor.extract_features(mesh)
    
    # Convert features to tensor
    feature_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Apply model
    with torch.no_grad():
        output = model(feature_tensor)
    
    # Convert output to numpy
    output = output.squeeze(0).cpu().numpy()
    
    # Create retopologized mesh from output
    retopologized_mesh = feature_extractor.create_mesh_from_features(output, mesh)
    
    return retopologized_mesh, features


def export_mesh(mesh: trimesh.Trimesh, 
               output_path: Union[str, Path], 
               format: str = "obj") -> None:
    """
    Export a mesh to a file.
    
    Args:
        mesh: Mesh to export
        output_path: Path to save the mesh
        format: Output format (obj, ply)
    """
    # Create output directory if needed
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Export mesh
    if format.lower() == "obj":
        mesh.export(output_path)
    elif format.lower() == "ply":
        mesh.export(output_path)
    else:
        raise ValueError(f"Unsupported format: {format}")


def process_directory(input_dir: Union[str, Path], 
                     model: torch.nn.Module, 
                     feature_extractor: FeatureExtractor,
                     config: Dict,
                     output_dir: Union[str, Path],
                     device: str,
                     format: str = "obj",
                     visualize: bool = False,
                     evaluate: bool = False) -> None:
    """
    Process all meshes in a directory.
    
    Args:
        input_dir: Path to input directory
        model: Trained model
        feature_extractor: Feature extractor
        config: Configuration dictionary
        output_dir: Path to output directory
        device: Device to use for inference
        format: Output format (obj, ply)
        visualize: Whether to generate visualizations
        evaluate: Whether to evaluate retopology quality
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize visualizer and evaluator if needed
    visualizer = MeshVisualizer(config) if visualize else None
    evaluator = MeshEvaluator(config) if evaluate else None
    
    # Get list of mesh files
    input_dir = Path(input_dir)
    mesh_files = []
    for ext in [".obj", ".ply", ".stl"]:
        mesh_files.extend(list(input_dir.glob(f"*{ext}")))
    
    # Process each mesh
    for mesh_path in tqdm(mesh_files, desc="Processing meshes"):
        # Create output path
        output_path = output_dir / f"{mesh_path.stem}_retopologized.{format}"
        
        # Process mesh
        retopologized_mesh, _ = process_mesh(mesh_path, model, feature_extractor, config, device)
        
        # Export mesh
        export_mesh(retopologized_mesh, output_path, format)
        
        # Generate visualizations if requested
        if visualize:
            vis_output_dir = output_dir / "visualizations" / mesh_path.stem
            visualizer.export_comparison(mesh_path, output_path, vis_output_dir)
        
        # Evaluate retopology quality if requested
        if evaluate:
            eval_output_dir = output_dir / "evaluation" / mesh_path.stem
            eval_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Evaluate meshes
            metrics = evaluator.evaluate(mesh_path, output_path)
            
            # Save metrics
            evaluator.save_metrics(eval_output_dir / "metrics.txt")
            
            # Print metrics
            print(f"Evaluation metrics for {mesh_path.name}:")
            evaluator.print_metrics()
    
    print(f"Processed {len(mesh_files)} meshes. Results saved to {output_dir}")


def main():
    """Main function."""
    args = parse_args()
    config = load_config(args.config)
    
    # Set device
    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.model, config, device)
    print(f"Loaded model from {args.model}")
    
    # Initialize feature extractor
    feature_extractor = FeatureExtractor(config)
    
    # Process input
    input_path = Path(args.input)
    if input_path.is_file():
        # Process single mesh
        retopologized_mesh, _ = process_mesh(input_path, model, feature_extractor, config, device)
        
        # Export mesh
        output_path = Path(args.output) / f"{input_path.stem}_retopologized.{args.format}"
        export_mesh(retopologized_mesh, output_path, args.format)
        
        # Generate visualizations if requested
        if args.visualize:
            vis_output_dir = Path(args.output) / "visualizations" / input_path.stem
            visualizer = MeshVisualizer(config)
            visualizer.export_comparison(input_path, output_path, vis_output_dir)
        
        # Evaluate retopology quality if requested
        if args.evaluate:
            eval_output_dir = Path(args.output) / "evaluation" / input_path.stem
            eval_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Evaluate meshes
            evaluator = MeshEvaluator(config)
            metrics = evaluator.evaluate(input_path, output_path)
            
            # Save metrics
            evaluator.save_metrics(eval_output_dir / "metrics.txt")
            
            # Print metrics
            print(f"Evaluation metrics for {input_path.name}:")
            evaluator.print_metrics()
        
        print(f"Processed {input_path.name}. Result saved to {output_path}")
    else:
        # Process directory
        process_directory(
            input_path, 
            model, 
            feature_extractor, 
            config, 
            args.output, 
            device, 
            args.format, 
            args.visualize, 
            args.evaluate
        )


if __name__ == "__main__":
    main() 