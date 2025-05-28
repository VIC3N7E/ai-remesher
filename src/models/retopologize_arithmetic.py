import trimesh
import numpy as np
import argparse
import os
from src.retopology_methods.quadric_error_method import quadric_error_method
from src.retopology_methods.voxel_remeshing import voxel_remeshing

def main(input_path: str, output_dir: str, method: str, **kwargs):
    """
    Main function for retopology.
    Args:
        input_path: Path to input mesh
        output_dir: Directory to save the result
        method: Retopology method to use ('qem', 'voxel', etc.)
        **kwargs: Additional arguments for the retopology method
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load input mesh
    mesh = trimesh.load(input_path, force='mesh')
    
    # Apply selected retopology method
    if method == 'qem':
        result = quadric_error_method(mesh, **kwargs)
    elif method == 'voxel':
        result = voxel_remeshing(mesh, **kwargs)
    else:
        raise ValueError(f"Unknown retopology method: {method}")
    
    # Save result in output directory
    output_path = os.path.join(output_dir, f"{method}_result.obj")
    result.export(output_path)
    
    print(f"Retopology completed. Result saved as {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Retopologize a mesh using various methods.")
    parser.add_argument('input', help='Path to input mesh file')
    parser.add_argument('output_dir', help='Directory to save the result')
    parser.add_argument('--method', choices=['qem', 'voxel'], default='qem',
                      help='Retopology method to use')
    
    # QEM method arguments
    parser.add_argument('--target-faces', type=int, default=None,
                      help='Target number of faces for QEM method')
    
    # Voxel method arguments
    parser.add_argument('--resolution', type=float, default=0.1,
                      help='Voxel resolution (voxel method)')
    parser.add_argument('--adaptive', action='store_true',
                      help='Use adaptive resolution (voxel method)')
    parser.add_argument('--curvature-threshold', type=float, default=0.1,
                      help='Curvature threshold for adaptive resolution (voxel method)')
    
    args = parser.parse_args()
    
    # Prepare kwargs based on selected method
    if args.method == 'qem':
        kwargs = {
            'target_faces': args.target_faces
        }
    else:  # voxel
        kwargs = {
            'resolution': args.resolution,
            'adaptive': args.adaptive,
            'curvature_threshold': args.curvature_threshold
        }
    
    main(args.input, args.output_dir, args.method, **kwargs) 