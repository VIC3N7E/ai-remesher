import os
import torch
import trimesh
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
import gc
import psutil
import time
import atexit
from retopology_model import (
    RetopologyModel, 
    retopologize_mesh, 
    compute_chamfer_distance, 
    compute_normal_loss,
    mesh_to_graph
)

def get_memory_usage():
    """Get current memory usage in GB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024 / 1024  # Convert to GB

def cleanup_memory():
    """Clean up memory by clearing cache and running garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def simplify_mesh(mesh: trimesh.Trimesh, target_faces: int = 5000) -> trimesh.Trimesh:
    """Simplify mesh to reduce memory usage."""
    if len(mesh.faces) > target_faces:
        return mesh.simplify_quadratic_decimation(target_faces)
    return mesh

def load_model(checkpoint_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu") -> RetopologyModel:
    """Load a trained model from checkpoint."""
    try:
        model = RetopologyModel().to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"Successfully loaded model from {checkpoint_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def evaluate_mesh(model: RetopologyModel, input_mesh: trimesh.Trimesh, target_mesh: trimesh.Trimesh, device: str, target_faces: int) -> dict:
    """Evaluate model performance on a single mesh pair."""
    try:
        # Simplify meshes to reduce memory usage
        input_mesh = simplify_mesh(input_mesh, target_faces)
        target_mesh = simplify_mesh(target_mesh, target_faces)
        
        # Convert meshes to graph format
        input_graph = mesh_to_graph(input_mesh)
        target_graph = mesh_to_graph(target_mesh)
        
        # Move to device
        input_graph = input_graph.to(device)
        target_graph = target_graph.to(device)
        
        # Get model prediction
        with torch.no_grad():
            output_vertices = model(input_graph)
            
            # Compute chamfer loss
            chamfer_loss = compute_chamfer_distance(output_vertices, target_graph.x)
            
            # For normal loss, we need to reconstruct faces
            # Use the input mesh's face structure for the output
            pred_faces = torch.tensor(input_mesh.faces, dtype=torch.long, device=device).t().contiguous()
            target_faces = torch.tensor(target_mesh.faces, dtype=torch.long, device=device).t().contiguous()
            
            # Compute normal loss
            normal_loss = compute_normal_loss(output_vertices, target_graph.x, pred_faces, target_faces)
            total_loss = chamfer_loss + 0.01 * normal_loss
            
            # Store results before cleanup
            result = {
                'chamfer_loss': chamfer_loss.item(),
                'normal_loss': normal_loss.item(),
                'total_loss': total_loss.item(),
                'input_vertices': len(input_mesh.vertices),
                'input_faces': len(input_mesh.faces),
                'output_vertices': len(output_vertices),
                'target_vertices': len(target_mesh.vertices),
                'target_faces': len(target_mesh.faces)
            }
            
            # Clean up tensors
            del input_graph, target_graph, output_vertices, pred_faces, target_faces
            cleanup_memory()
            
            return result
            
    except Exception as e:
        print(f"Error evaluating mesh: {str(e)}")
        # Clean up any remaining tensors
        cleanup_memory()
        raise

def evaluate_dataset(model: RetopologyModel, dataset_dir: str, split: str, device: str, max_models: int = None, target_faces: int = 5000) -> dict:
    """Evaluate model on all meshes in a dataset split."""
    results = []
    modified_dir = Path(dataset_dir) / 'modified' / split
    original_dir = Path(dataset_dir) / 'original' / split
    
    # Get all mesh pairs
    mesh_pairs = []
    for mesh_file in modified_dir.glob('*_modified.obj'):
        original_file = original_dir / mesh_file.name.replace('_modified', '_original')
        if original_file.exists():
            mesh_pairs.append((mesh_file, original_file))
    
    # Limit number of models if specified
    if max_models is not None:
        mesh_pairs = mesh_pairs[:max_models]
    
    print(f"\nEvaluating {len(mesh_pairs)} meshes in {split} split...")
    print(f"Initial memory usage: {get_memory_usage():.2f} GB")
    
    # Process one mesh at a time
    for i, (modified_path, original_path) in enumerate(tqdm(mesh_pairs, desc=f"Processing {split} meshes")):
        try:
            print(f"\nProcessing mesh {i+1}/{len(mesh_pairs)}")
            print(f"Memory usage before mesh: {get_memory_usage():.2f} GB")
            
            # Load meshes
            input_mesh = trimesh.load(modified_path)
            target_mesh = trimesh.load(original_path)
            
            # Evaluate
            result = evaluate_mesh(model, input_mesh, target_mesh, device, target_faces)
            result['mesh_name'] = modified_path.stem
            results.append(result)
            
            # Clean up meshes
            del input_mesh, target_mesh
            cleanup_memory()
            
            print(f"Memory usage after mesh: {get_memory_usage():.2f} GB")
            
        except Exception as e:
            print(f"Error processing {modified_path}: {str(e)}")
            # Clean up any remaining resources
            cleanup_memory()
            continue
    
    # Compute average metrics
    if results:
        avg_metrics = {
            'chamfer_loss': np.mean([r['chamfer_loss'] for r in results]),
            'normal_loss': np.mean([r['normal_loss'] for r in results]),
            'total_loss': np.mean([r['total_loss'] for r in results]),
            'avg_input_vertices': np.mean([r['input_vertices'] for r in results]),
            'avg_output_vertices': np.mean([r['output_vertices'] for r in results]),
            'avg_target_vertices': np.mean([r['target_vertices'] for r in results])
        }
        return avg_metrics
    return None

def main():
    # Register cleanup function
    atexit.register(cleanup_memory)
    
    parser = argparse.ArgumentParser(description="Evaluate model performance and check for overfitting")
    parser.add_argument('model_path', help='Path to the trained model checkpoint')
    parser.add_argument('dataset_dir', help='Path to the dataset directory')
    parser.add_argument('--max-models', type=int, help='Maximum number of models to evaluate per split')
    parser.add_argument('--target-faces', type=int, default=5000, help='Target number of faces for mesh simplification')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.model_path, device)
    
    # Evaluate on both splits
    train_metrics = evaluate_dataset(model, args.dataset_dir, 'train', device, args.max_models, args.target_faces)
    val_metrics = evaluate_dataset(model, args.dataset_dir, 'validation', device, args.max_models, args.target_faces)
    
    # Print results
    print("\nEvaluation Results:")
    print("\nTraining Set Metrics:")
    for metric, value in train_metrics.items():
        print(f"{metric}: {value:.6f}")
    
    print("\nValidation Set Metrics:")
    for metric, value in val_metrics.items():
        print(f"{metric}: {value:.6f}")
    
    # Check for overfitting/underfitting
    print("\nOverfitting/Underfitting Analysis:")
    chamfer_diff = abs(train_metrics['chamfer_loss'] - val_metrics['chamfer_loss'])
    normal_diff = abs(train_metrics['normal_loss'] - val_metrics['normal_loss'])
    total_diff = abs(train_metrics['total_loss'] - val_metrics['total_loss'])
    
    print(f"Chamfer Loss Difference: {chamfer_diff:.6f}")
    print(f"Normal Loss Difference: {normal_diff:.6f}")
    print(f"Total Loss Difference: {total_diff:.6f}")
    
    # Analysis
    if total_diff < 0.01:
        print("\nThe model appears to be well-fitted (good generalization)")
    elif train_metrics['total_loss'] < val_metrics['total_loss']:
        print("\nThe model shows signs of overfitting (performs better on training than validation)")
    else:
        print("\nThe model shows signs of underfitting (performs poorly on both sets)")

if __name__ == '__main__':
    main() 