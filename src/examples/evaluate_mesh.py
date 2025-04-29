#!/usr/bin/env python3
"""
Example script to demonstrate how to use the evaluation metrics and visualization tools.
"""

import argparse
import os
import yaml
import trimesh
from pathlib import Path
from typing import Union, List, Dict

from src.utils.evaluation import MeshEvaluator
from src.utils.visualization import MeshVisualizer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate and visualize mesh retopology results.")
    parser.add_argument("--original", type=str, required=True, help="Path to the original mesh file")
    parser.add_argument("--retopologized", type=str, required=True, help="Path to the retopologized mesh file")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to the configuration file")
    parser.add_argument("--output", type=str, default="output/evaluation", help="Path to the output directory")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load the configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def evaluate_mesh(original_mesh_path: Union[str, Path], 
                 retopologized_mesh_path: Union[str, Path], 
                 config: Dict, 
                 output_dir: Union[str, Path]) -> Dict[str, float]:
    """
    Evaluate the quality of a retopologized mesh compared to the original mesh.
    
    Args:
        original_mesh_path: Path to the original mesh file
        retopologized_mesh_path: Path to the retopologized mesh file
        config: Configuration dictionary
        output_dir: Path to the output directory
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize evaluator
    evaluator = MeshEvaluator(config)
    
    # Evaluate meshes
    metrics = evaluator.evaluate(original_mesh_path, retopologized_mesh_path)
    
    # Save metrics
    evaluator.save_metrics(output_dir / "metrics.txt")
    
    # Print metrics
    evaluator.print_metrics()
    
    return metrics


def visualize_mesh(original_mesh_path: Union[str, Path], 
                  retopologized_mesh_path: Union[str, Path], 
                  config: Dict, 
                  output_dir: Union[str, Path]) -> None:
    """
    Visualize a comparison between original and retopologized meshes.
    
    Args:
        original_mesh_path: Path to the original mesh file
        retopologized_mesh_path: Path to the retopologized mesh file
        config: Configuration dictionary
        output_dir: Path to the output directory
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize visualizer
    visualizer = MeshVisualizer(config)
    
    # Export comparison
    visualizer.export_comparison(original_mesh_path, retopologized_mesh_path, output_dir)


def main():
    """Main function."""
    args = parse_args()
    config = load_config(args.config)
    
    # Evaluate meshes
    metrics = evaluate_mesh(args.original, args.retopologized, config, args.output)
    
    # Visualize meshes if requested
    if args.visualize:
        visualize_mesh(args.original, args.retopologized, config, args.output)
    
    print(f"Evaluation completed. Results saved to {args.output}")


if __name__ == "__main__":
    main() 