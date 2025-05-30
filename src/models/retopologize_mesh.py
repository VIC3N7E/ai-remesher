import os
import torch
import trimesh
import argparse
from pathlib import Path
from retopology_model import RetopologyModel, mesh_to_graph
import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

def retopologize_mesh(model: RetopologyModel, input_mesh: trimesh.Trimesh, device: str) -> trimesh.Trimesh:
    """Retopologize a mesh using the trained model."""
    model.eval()
    
    # Convert input mesh to graph
    input_data = mesh_to_graph(input_mesh)
    input_data = input_data.to(device)
    
    # Get prediction
    with torch.no_grad():
        pred_vertices = model(input_data)
    
    # Create new mesh with predicted vertices
    retopologized_mesh = trimesh.Trimesh(
        vertices=pred_vertices.cpu().numpy(),
        faces=input_mesh.faces
    )
    
    return retopologized_mesh

def main():
    parser = argparse.ArgumentParser(description='Retopologize a mesh using a trained model')
    parser.add_argument('checkpoint_path', type=str, help='Path to the model checkpoint')
    parser.add_argument('input_mesh', type=str, help='Path to the input mesh file')
    parser.add_argument('output_mesh', type=str, help='Path to save the retopologized mesh')
    parser.add_argument('--hidden-channels', type=int, default=32, help='Number of hidden channels in the model')
    parser.add_argument('--num-layers', type=int, default=3, help='Number of layers in encoder and decoder')
    args = parser.parse_args()
    
    setup_logging()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Load model
    model = RetopologyModel(
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    logging.info(f"Loaded checkpoint from {args.checkpoint_path}")
    
    # Load input mesh
    input_mesh = trimesh.load(args.input_mesh)
    logging.info(f"Loaded input mesh with {len(input_mesh.vertices)} vertices and {len(input_mesh.faces)} faces")
    
    # Retopologize mesh
    retopologized_mesh = retopologize_mesh(model, input_mesh, device)
    logging.info(f"Retopologized mesh has {len(retopologized_mesh.vertices)} vertices and {len(retopologized_mesh.faces)} faces")
    
    # Save retopologized mesh
    retopologized_mesh.export(args.output_mesh)
    logging.info(f"Saved retopologized mesh to {args.output_mesh}")

if __name__ == '__main__':
    main() 