import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import yaml
from tqdm import tqdm
import logging
from typing import Dict, Optional
import os
import argparse
from torch.utils.data import DataLoader

from models.retopology_model import RetopologyModel
from data.mesh_processor import MeshProcessor
from data.dataset import MeshDataset

def setup_logging(config: Dict):
    """Setup logging configuration."""
    log_dir = Path('logs')
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'train.log'),
            logging.StreamHandler()
        ]
    )

def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    loss: float,
    checkpoint_dir: Path
):
    """Save model checkpoint."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, checkpoint_path)
    
    logging.info(f'Saved checkpoint to {checkpoint_path}')

def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    print("Starting training epoch...")
    for batch_idx, batch in enumerate(tqdm(dataloader, desc='Training')):
        print(f"\nProcessing batch {batch_idx+1}...")
        # Move batch to device
        batch = {
            k: {
                k2: v2.to(device) if isinstance(v2, torch.Tensor) else v2
                for k2, v2 in v.items()
            } if isinstance(v, dict) else (
                v.to(device) if isinstance(v, torch.Tensor) else v
            )
            for k, v in batch.items()
        }
        print("Batch moved to device")
        
        # Forward pass
        print("Running forward pass...")
        positions = batch['vertex_features']['positions']
        normals = batch['vertex_features'].get('normals', torch.zeros_like(positions))
        
        # Ensure all tensors are 2D
        if positions.dim() == 3:
            positions = positions.reshape(-1, positions.shape[-1])
        if normals.dim() == 3:
            normals = normals.reshape(-1, normals.shape[-1])
        
        vertex_features = torch.cat([positions, normals], dim=1)
        edge_index = faces_to_edges(batch['faces'])
        
        predictions = model(
            vertex_features,
            edge_index,
            None  # No batch information for single mesh
        )
        print("Forward pass completed")
        
        # Compute loss
        print("Computing loss...")
        loss_dict = model.compute_loss(predictions, batch)
        loss = loss_dict['total_loss']
        print(f"Loss computed: {loss.item():.6f}")
        
        # Backward pass
        print("Running backward pass...")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("Backward pass completed")
        
        # Display loss for each iteration
        print(f"Batch {batch_idx+1}/{len(dataloader)} - Loss: {loss.item():.6f}")
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device
) -> float:
    """Validate the model."""
    model.eval()
    total_loss = 0
    
    print("Starting validation...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc='Validation')):
            print(f"\nProcessing validation batch {batch_idx+1}...")
            # Move batch to device
            batch = {
                k: {
                    k2: v2.to(device) if isinstance(v2, torch.Tensor) else v2
                    for k2, v2 in v.items()
                } if isinstance(v, dict) else (
                    v.to(device) if isinstance(v, torch.Tensor) else v
                )
                for k, v in batch.items()
            }
            print("Validation batch moved to device")
            
            # Forward pass
            print("Running validation forward pass...")
            positions = batch['vertex_features']['positions']
            normals = batch['vertex_features'].get('normals', torch.zeros_like(positions))
            
            # Ensure all tensors are 2D
            if positions.dim() == 3:
                positions = positions.reshape(-1, positions.shape[-1])
            if normals.dim() == 3:
                normals = normals.reshape(-1, normals.shape[-1])
            
            vertex_features = torch.cat([positions, normals], dim=1)
            edge_index = faces_to_edges(batch['faces'])
            
            predictions = model(
                vertex_features,
                edge_index,
                None  # No batch information for single mesh
            )
            print("Validation forward pass completed")
            
            # Compute loss
            print("Computing validation loss...")
            loss_dict = model.compute_loss(predictions, batch)
            loss = loss_dict['total_loss']
            print(f"Validation loss computed: {loss.item():.6f}")
            
            # Display loss for each iteration
            print(f"Validation Batch {batch_idx+1}/{len(dataloader)} - Loss: {loss.item():.6f}")
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def faces_to_edges(faces: torch.Tensor) -> torch.Tensor:
    """Convert face indices to edge indices.
    
    Args:
        faces: Tensor of face indices with shape (num_faces, 3)
        
    Returns:
        edge_index: Tensor of edge indices with shape (2, num_edges)
    """
    # Create edges from faces
    edges = torch.cat([
        torch.stack([faces[:, i], faces[:, (i + 1) % 3]], dim=1)
        for i in range(3)
    ], dim=0)
    
    # Remove duplicate edges
    edges = torch.unique(torch.sort(edges, dim=1)[0], dim=0)
    
    # Convert to edge index format (2, num_edges)
    edge_index = edges.t().contiguous()
    
    return edge_index

def main():
    """Main training function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train the mesh processing model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Set up logging
    setup_logging(config)
    
    # Set device
    device = torch.device(config['training']['device'])
    
    # Create mesh processor
    processor = MeshProcessor(config)
    
    # Create datasets
    train_dataset = MeshDataset(
        data_dir=config['data']['raw_dir'],
        processor=processor,
        split='train'
    )
    val_dataset = MeshDataset(
        data_dir=config['data']['raw_dir'],
        processor=processor,
        split='val'
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    model = RetopologyModel(
        in_channels=config['model']['in_channels'],
        hidden_channels=config['model']['hidden_channels'],
        num_transformer_layers=config['model']['num_transformer_layers']
    ).to(device)
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Create scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"Starting training for {config['training']['num_epochs']} epochs...")
    print(f"Training on device: {device}")
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Learning rate: {config['training']['learning_rate']}")
    
    for epoch in range(config['training']['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['training']['num_epochs']}")
        print("-" * 50)
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)
        print(f"Epoch {epoch+1} - Training Loss: {train_loss:.6f}")
        
        # Validate
        val_loss = validate(model, val_loader, device)
        print(f"Epoch {epoch+1} - Validation Loss: {val_loss:.6f}")
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save checkpoint if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                val_loss,
                os.path.join(args.checkpoint_dir, 'best_model.pth')
            )
            print(f"Saved checkpoint with validation loss: {val_loss:.6f}")
        else:
            patience_counter += 1
            print(f"No improvement in validation loss. Patience: {patience_counter}/{config['training']['early_stopping_patience']}")
        
        # Early stopping
        if patience_counter >= config['training']['early_stopping_patience']:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")

if __name__ == '__main__':
    main() 