#!/usr/bin/env python3
"""
Example script to demonstrate how to use the mesh processing pipeline with the configuration file.
"""

import argparse
import os
import yaml
import torch
from pathlib import Path
from typing import Union, List

from src.data.mesh_dataset import MeshDataset
from src.data.mesh_loader import MeshLoader
from src.models.mesh_net import MeshNet
from src.utils.mesh_utils import load_mesh, save_mesh


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Process a mesh using the AI Remesher pipeline.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input mesh file or directory")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to the configuration file")
    parser.add_argument("--output", type=str, help="Path to the output directory")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for processing")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load the configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def process_mesh(mesh_path: Union[str, Path], config: dict, output_dir: Union[str, Path] = None) -> None:
    """Process a single mesh file."""
    # Load the mesh
    mesh = load_mesh(mesh_path)
    
    # Create a dataset
    dataset = MeshDataset([mesh], config)
    
    # Create a data loader
    loader = MeshLoader(dataset, batch_size=config["data"]["batch_size"], num_workers=config["data"]["num_workers"])
    
    # Create the model
    model = MeshNet(config["model"])
    
    # Process the mesh
    for batch in loader:
        # Move the batch to the device
        batch = {k: v.to(model.device) for k, v in batch.items()}
        
        # Forward pass
        output = model(batch)
        
        # Save the processed mesh
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{Path(mesh_path).stem}_processed.obj"
            save_mesh(output, output_path)


def process_directory(directory_path: Union[str, Path], config: dict, output_dir: Union[str, Path] = None) -> None:
    """Process all meshes in a directory."""
    directory_path = Path(directory_path)
    mesh_files = list(directory_path.glob("*.obj"))
    
    for mesh_file in mesh_files:
        process_mesh(mesh_file, config, output_dir)


def main():
    """Main function."""
    args = parse_args()
    config = load_config(args.config)
    
    # Override config with command line arguments
    config["data"]["batch_size"] = args.batch_size
    config["data"]["num_workers"] = args.num_workers
    
    input_path = Path(args.input)
    if input_path.is_file():
        process_mesh(input_path, config, args.output)
    elif input_path.is_dir():
        process_directory(input_path, config, args.output)
    else:
        raise ValueError(f"Input path {input_path} does not exist.")


if __name__ == "__main__":
    main() 