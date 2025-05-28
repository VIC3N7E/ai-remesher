import os
import torch
import trimesh
import argparse
from pathlib import Path
from retopology_model import RetopologyModel, retopologize_mesh

def load_model(checkpoint_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu") -> RetopologyModel:
    """Load a trained model from checkpoint."""
    try:
        model = RetopologyModel().to(device)
        
        # Load checkpoint with weights_only=True for security
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"Successfully loaded model from {checkpoint_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def process_mesh(model: RetopologyModel, input_path: str, output_path: str):
    """Process a single mesh using the trained model."""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Load input mesh
        print(f"Loading input mesh from {input_path}")
        input_mesh = trimesh.load(input_path)
        
        # Retopologize mesh
        print("Retopologizing mesh...")
        retopologized_mesh = retopologize_mesh(model, input_mesh)
        
        # Save result
        retopologized_mesh.export(output_path)
        print(f"Successfully saved retopologized mesh to: {output_path}")
        
        # Print mesh statistics
        print("\nMesh Statistics:")
        print(f"Input vertices: {len(input_mesh.vertices)}")
        print(f"Input faces: {len(input_mesh.faces)}")
        print(f"Output vertices: {len(retopologized_mesh.vertices)}")
        print(f"Output faces: {len(retopologized_mesh.faces)}")
        
    except Exception as e:
        print(f"Error processing mesh: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Retopologize meshes using trained model")
    parser.add_argument('model_path', help='Path to the trained model checkpoint')
    parser.add_argument('input_path', help='Path to input mesh file')
    parser.add_argument('output_path', help='Path to save retopologized mesh')
    
    args = parser.parse_args()
    
    # Verify input file exists
    if not os.path.exists(args.input_path):
        print(f"Error: Input file {args.input_path} does not exist")
        return
    
    # Verify model file exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file {args.model_path} does not exist")
        return
    
    try:
        # Load model
        model = load_model(args.model_path)
        
        # Process mesh
        process_mesh(model, args.input_path, args.output_path)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return

if __name__ == '__main__':
    main() 