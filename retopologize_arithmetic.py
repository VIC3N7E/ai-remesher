import trimesh
import numpy as np
import argparse
import os
from retopology_methods.quadric_error_method import quad_based_retopology
from src.retopology_methods.voxel_remeshing import voxel_remeshing
from src.utils.checkpoint_saver import CheckpointSaver

def main(input_path: str, output_dir: str, method: str, **kwargs):
    """
    Main function for retopology with checkpoint saving.
    Args:
        input_path: Path to input mesh
        output_dir: Directory to save checkpoints
        method: Retopology method to use ('quad', 'voxel', etc.)
        **kwargs: Additional arguments for the retopology method
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load input mesh
    mesh = trimesh.load(input_path, force='mesh')
    
    # Initialize checkpoint saver
    saver = CheckpointSaver(output_dir)
    
    # Save initial mesh
    saver.save_checkpoint(mesh, "initial", "input_mesh")
    
    # Apply selected retopology method
    if method == 'quad':
        result = quad_based_retopology(mesh, **kwargs)
        saver.save_checkpoint(result, "quad", "final_result")
    elif method == 'voxel':
        result = voxel_remeshing(mesh, **kwargs)
        saver.save_checkpoint(result, "voxel", "final_result")
    else:
        raise ValueError(f"Unknown retopology method: {method}")
    
    print(f"Retopology completed. Checkpoints saved in {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Retopologize a mesh using various methods.")
    parser.add_argument('input', help='Path to input mesh file')
    parser.add_argument('output_dir', help='Directory to save checkpoints')
    parser.add_argument('--method', choices=['quad', 'voxel'], default='quad',
                      help='Retopology method to use')
    
    # Quad-based method arguments
    parser.add_argument('--iterations', type=int, default=5,
                      help='Number of optimization iterations (quad method)')
    parser.add_argument('--feature-threshold', type=float, default=np.pi/4,
                      help='Feature edge detection threshold (quad method)')
    
    # Voxel method arguments
    parser.add_argument('--resolution', type=float, default=0.1,
                      help='Voxel resolution (voxel method)')
    parser.add_argument('--adaptive', action='store_true',
                      help='Use adaptive resolution (voxel method)')
    parser.add_argument('--curvature-threshold', type=float, default=0.1,
                      help='Curvature threshold for adaptive resolution (voxel method)')
    
    args = parser.parse_args()
    
    # Prepare kwargs based on selected method
    if args.method == 'quad':
        kwargs = {
            'iterations': args.iterations,
            'feature_edges': None  # Will be computed based on threshold
        }
    else:  # voxel
        kwargs = {
            'resolution': args.resolution,
            'adaptive': args.adaptive,
            'curvature_threshold': args.curvature_threshold
        }
    
    main(args.input, args.output_dir, args.method, **kwargs) 