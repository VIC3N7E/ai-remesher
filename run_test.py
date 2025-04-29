#!/usr/bin/env python3
"""
Script to run the mesh processing test from the project root directory.
"""

import os
import sys
from pathlib import Path

# Add the current directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Import and run the test function
from src.examples.test_pipeline import test_mesh_processing

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the mesh processing pipeline.")
    parser.add_argument("--mesh", type=str, required=True, help="Path to input mesh file")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to configuration file")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    
    args = parser.parse_args()
    
    test_mesh_processing(
        mesh_path=args.mesh,
        config_path=args.config,
        output_dir=args.output
    ) 