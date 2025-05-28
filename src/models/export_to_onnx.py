import torch
import argparse
from retopology_model import RetopologyModel
import os

def export_to_onnx(checkpoint_path: str, output_path: str):
    """Export the model to ONNX format."""
    # Load the model
    model = RetopologyModel()
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create dummy input
    # Assuming input mesh has 1000 vertices and 3 coordinates (x,y,z)
    dummy_vertices = torch.randn(1000, 3)
    dummy_faces = torch.randint(0, 1000, (2000, 3))  # Assuming 2000 faces
    
    # Create edge index from faces
    edge_index = torch.stack([
        dummy_faces[:, [0, 1, 1, 2, 2, 0]].flatten(),
        dummy_faces[:, [1, 0, 2, 1, 0, 2]].flatten()
    ])
    edge_index = torch.unique(edge_index, dim=1)
    
    # Create face index
    face_index = dummy_faces.t().contiguous()
    
    # Create dummy input data
    dummy_input = {
        'x': dummy_vertices,
        'edge_index': edge_index,
        'face_index': face_index
    }
    
    # Export to ONNX
    torch.onnx.export(
        model,
        (dummy_input,),
        output_path,
        input_names=['vertices', 'edge_index', 'face_index'],
        output_names=['output_vertices'],
        dynamic_axes={
            'vertices': {0: 'num_vertices'},
            'edge_index': {1: 'num_edges'},
            'face_index': {1: 'num_faces'},
            'output_vertices': {0: 'num_vertices'}
        },
        opset_version=11
    )
    
    print(f"Model exported to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument('checkpoint_path', help='Path to the model checkpoint')
    parser.add_argument('--output-path', help='Path to save the ONNX model', default='model.onnx')
    
    args = parser.parse_args()
    
    export_to_onnx(args.checkpoint_path, args.output_path)

if __name__ == '__main__':
    main() 