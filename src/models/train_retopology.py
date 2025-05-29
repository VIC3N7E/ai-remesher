import os
import torch
import trimesh
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple
import logging
from datetime import datetime
from retopology_model import RetopologyModel, RetopologyTrainer, mesh_to_graph, compute_chamfer_distance, compute_normal_loss
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
import torch.multiprocessing as mp
from functools import partial
from torch_geometric.data import Batch
import concurrent.futures
from typing import Dict
import torch.cuda.amp as amp

# Configurar logging
def setup_logging(dataset_dir: str):
    log_dir = os.path.join(dataset_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_dir

def load_mesh_pair(modified_path: str, original_path: str) -> Tuple[trimesh.Trimesh, trimesh.Trimesh]:
    """Load a pair of meshes (modified and original)."""
    modified_mesh = trimesh.load(modified_path)
    original_mesh = trimesh.load(original_path)
    return modified_mesh, original_mesh

def get_mesh_pairs(dataset_dir: str) -> List[Tuple[str, str]]:
    """Get all pairs of meshes from the dataset directory."""
    modified_dir = os.path.join(dataset_dir, "modified")
    original_dir = os.path.join(dataset_dir, "original")
    
    # Verificar se os diretórios existem
    if not os.path.exists(dataset_dir):
        raise ValueError(f"Dataset directory not found: {dataset_dir}")
    
    if not os.path.exists(modified_dir):
        raise ValueError(f"Modified directory not found: {modified_dir}")
    
    if not os.path.exists(original_dir):
        raise ValueError(f"Original directory not found: {original_dir}")
    
    # Listar arquivos em cada diretório
    modified_files = []
    original_files = []
    
    for root, _, files in os.walk(modified_dir):
        for file in files:
            if file.endswith("_modified.obj"):
                modified_files.append(os.path.join(root, file))
    
    for root, _, files in os.walk(original_dir):
        for file in files:
            if file.endswith("_original.obj"):
                original_files.append(os.path.join(root, file))
    
    # Verificar se encontrou arquivos
    if not modified_files:
        raise ValueError(f"No modified mesh files (*_modified.obj) found in {modified_dir}")
    
    if not original_files:
        raise ValueError(f"No original mesh files (*_original.obj) found in {original_dir}")
    
    # Criar pares de meshes
    mesh_pairs = []
    for modified_path in modified_files:
        # Obter o nome base do arquivo
        base_name = os.path.basename(modified_path).replace("_modified.obj", "")
        
        # Obter o caminho relativo
        rel_path = os.path.relpath(os.path.dirname(modified_path), modified_dir)
        
        # Construir o caminho do arquivo original correspondente
        original_path = os.path.join(original_dir, rel_path, f"{base_name}_original.obj")
        
        if os.path.exists(original_path):
            mesh_pairs.append((modified_path, original_path))
        else:
            logging.warning(f"Original mesh not found for {modified_path}: {original_path}")
    
    if not mesh_pairs:
        raise ValueError(
            f"No valid mesh pairs found. Please check:\n"
            f"1. File naming convention: *_modified.obj and *_original.obj\n"
            f"2. Directory structure: modified/ and original/ should have matching subdirectories\n"
            f"3. Found {len(modified_files)} modified files and {len(original_files)} original files"
        )
    
    logging.info(f"Found {len(mesh_pairs)} valid mesh pairs")
    logging.info(f"First pair: {mesh_pairs[0]}")
    
    return mesh_pairs

def custom_collate(batch):
    """Custom collate function for graph data"""
    modified_graphs, original_graphs = zip(*batch)
    return Batch.from_data_list(modified_graphs), Batch.from_data_list(original_graphs)

def process_mesh_pair(args: Tuple[Path, Path]) -> Dict:
    """Process a single mesh pair in parallel"""
    modified_path, original_path = args
    try:
        # Load and preprocess modified mesh with memory management
        modified_mesh = trimesh.load(modified_path)
        
        # Only process if necessary
        if not modified_mesh.is_watertight:
            modified_mesh.fill_holes()
        if len(modified_mesh.faces) > 0 and modified_mesh.faces.shape[1] > 3:
            modified_mesh = modified_mesh.triangulate()
        
        # Convert to graph format immediately to free memory
        modified_graph = mesh_to_graph(modified_mesh)
        del modified_mesh
        
        # Load and preprocess original mesh with memory management
        original_mesh = trimesh.load(original_path)
        
        # Only process if necessary
        if not original_mesh.is_watertight:
            original_mesh.fill_holes()
        if len(original_mesh.faces) > 0 and original_mesh.faces.shape[1] > 3:
            original_mesh = original_mesh.triangulate()
        
        # Convert to graph format
        original_graph = mesh_to_graph(original_mesh)
        del original_mesh
        
        # Force garbage collection
        import gc
        gc.collect()
        
        return {
            'path': str(modified_path),
            'modified': modified_graph,
            'original': original_graph,
            'success': True
        }
    except Exception as e:
        logging.error(f"Error preprocessing {modified_path}: {str(e)}")
        return {
            'path': str(modified_path),
            'success': False,
            'error': str(e)
        }

class MeshDataset(Dataset):
    def __init__(self, data_dir, models_per_epoch=None, num_workers=1, is_train=True):  # Reduced to single worker
        self.data_dir = Path(data_dir)
        self.models_per_epoch = models_per_epoch
        
        # Find all mesh pairs
        self.mesh_pairs = []
        if is_train:
            modified_dir = self.data_dir / 'modified' / 'train' 
            original_dir = self.data_dir / 'original' / 'train'
        else:
            modified_dir = self.data_dir / 'modified' / 'validation'
            original_dir = self.data_dir / 'original' / 'validation'
        
        logging.info(f"Searching for mesh pairs in:")
        logging.info(f"Modified directory: {modified_dir}")
        logging.info(f"Original directory: {original_dir}")
        
        modified_files = list(modified_dir.glob('*_modified.obj'))
        logging.info(f"Found {len(modified_files)} modified mesh files")
        
        for mesh_file in modified_files:
            # Get the base name (e.g., 'bed_0384' from 'bed_0384_modified.obj')
            base_name = mesh_file.stem.replace('_modified', '')
            original_file = original_dir / f"{base_name}_original.obj"
            
            if original_file.exists():
                self.mesh_pairs.append((mesh_file, original_file))
                logging.debug(f"Found pair: {mesh_file.name} <-> {original_file.name}")
            else:
                logging.warning(f"Original mesh not found for {mesh_file.name}: {original_file}")
        
        if not self.mesh_pairs:
            raise ValueError(
                f"No valid mesh pairs found in {data_dir}. Please check:\n"
                f"1. Directory structure: modified/ and original/ directories should exist\n"
                f"2. File naming convention: *_modified.obj and *_original.obj\n"
                f"3. Current directory structure:\n"
                f"   - Modified files: {list(modified_dir.glob('*_modified.obj'))}\n"
                f"   - Original files: {list(original_dir.glob('*_original.obj'))}"
            )
        
        # Shuffle the mesh pairs
        np.random.shuffle(self.mesh_pairs)
        
        # Limit the number of models if specified
        if models_per_epoch is not None:
            self.mesh_pairs = self.mesh_pairs[:models_per_epoch]
            
        logging.info(f"Using {len(self.mesh_pairs)} mesh pairs for training")
        if self.mesh_pairs:
            logging.info(f"First pair: {self.mesh_pairs[0]}")
            
        # Preprocess and cache meshes in parallel
        self.cached_meshes = {}
        self._preprocess_meshes_parallel(num_workers)
        
        # Clear any remaining memory
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    def _preprocess_meshes_parallel(self, num_workers: int):
        """Preprocess and cache all meshes in parallel"""
        logging.info(f"Preprocessing and caching meshes using {num_workers} workers...")
        
        # Process one mesh at a time to avoid memory issues
        for i, mesh_pair in enumerate(self.mesh_pairs):
            result = process_mesh_pair(mesh_pair)
            if result['success']:
                self.cached_meshes[result['path']] = {
                    'modified': result['modified'],
                    'original': result['original']
                }
            
            # Clear memory after each mesh
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            import gc
            gc.collect()
            
            # Log progress
            logging.info(f"Processed mesh {i+1}/{len(self.mesh_pairs)}")

    def __len__(self):
        return len(self.mesh_pairs)
    
    def __getitem__(self, idx):
        modified_path, original_path = self.mesh_pairs[idx]
        cached_data = self.cached_meshes[str(modified_path)]
        return cached_data['modified'], cached_data['original']

def main():
    parser = argparse.ArgumentParser(description='Train mesh retopology model')
    parser.add_argument('data_dir', type=str, help='Directory containing mesh pairs')
    parser.add_argument('--num-epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for training')  # Reduced to 1
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--checkpoint-interval', type=int, default=10, help='Save checkpoint every N batches')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--num-workers', type=int, default=1, help='Number of workers for data loading')  # Reduced to 1
    parser.add_argument('--hidden-channels', type=int, default=128, help='Number of hidden channels in the model')
    parser.add_argument('--num-layers', type=int, default=3, help='Number of layers in encoder and decoder')
    parser.add_argument('--models-per-epoch', type=int, help='Number of models to use per epoch (default: use all available models)')
    parser.add_argument('--is-train', type=bool, default=True, help='Whether to use training or validation data')
    args = parser.parse_args()
    
    # Create checkpoint and logs directories if they don't exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join('logs', 'training.log')),
            logging.StreamHandler()
        ]
    )
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Create model
    model = RetopologyModel(
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers
    ).to(device)
    
    # Create trainer
    trainer = RetopologyTrainer(model, learning_rate=args.learning_rate)
    
    # Create dataset
    dataset = MeshDataset(args.data_dir, models_per_epoch=args.models_per_epoch, is_train=True)
    
    # Create data loader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    # Training loop
    for epoch in range(args.num_epochs):
        logging.info(f"Starting epoch {epoch + 1}/{args.num_epochs}")
        
        for batch_idx, (input_mesh, target_mesh) in enumerate(dataloader):
            try:
                # Move to device
                input_mesh = input_mesh.to(device)
                target_mesh = target_mesh.to(device)
                
                # Training step
                chamfer_loss, normal_loss = trainer.train_step(input_mesh, target_mesh)
                
                # Log progress
                if (batch_idx + 1) % 10 == 0:
                    logging.info(f"Epoch {epoch + 1}, Batch {batch_idx + 1}: "
                               f"Chamfer Loss = {chamfer_loss:.6f}, "
                               f"Normal Loss = {normal_loss:.6f}")
                
                # Save checkpoint
                if (batch_idx + 1) % args.checkpoint_interval == 0:
                    checkpoint_path = os.path.join(
                        args.checkpoint_dir,
                        f'checkpoint_epoch_{epoch + 1}_batch_{batch_idx + 1}.pt'
                    )
                    torch.save({
                        'epoch': epoch + 1,
                        'batch': batch_idx + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': trainer.optimizer.state_dict(),
                        'chamfer_loss': chamfer_loss,
                        'normal_loss': normal_loss
                    }, checkpoint_path)
                    logging.info(f"Saved checkpoint to {checkpoint_path}")
                
            except Exception as e:
                logging.error(f"Error in batch {batch_idx + 1}: {str(e)}")
                continue
        
        logging.info(f"Completed epoch {epoch + 1}")

if __name__ == '__main__':
    main() 