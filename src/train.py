import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import yaml
from tqdm import tqdm
import logging
from typing import Dict, Optional

from models.retopology_model import RetopologyModel
from data.mesh_processor import MeshProcessor
from data.dataset import MeshDataModule

def setup_logging(log_dir: Path):
    """Setup logging configuration."""
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
    
    for batch in tqdm(dataloader, desc='Training'):
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Forward pass
        predictions = model(
            batch['vertex_features'],
            batch['edge_index'],
            batch.get('batch')
        )
        
        # Compute loss
        loss_dict = model.compute_loss(predictions, batch)
        loss = loss_dict['total_loss']
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
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
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            predictions = model(
                batch['vertex_features'],
                batch['edge_index'],
                batch.get('batch')
            )
            
            # Compute loss
            loss_dict = model.compute_loss(predictions, batch)
            loss = loss_dict['total_loss']
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def main(
    config_path: str = 'src/configs/default.yaml',
    checkpoint_dir: Optional[str] = None
):
    # Load configuration
    config = load_config(config_path)
    
    # Setup logging
    log_dir = Path('logs')
    setup_logging(log_dir)
    
    # Setup device
    device = torch.device(config['training']['device'])
    
    # Create mesh processor
    processor = MeshProcessor(
        max_vertices=config['preprocessing']['max_vertices'],
        normalize=config['preprocessing']['normalize'],
        compute_normals=config['preprocessing']['compute_normals'],
        compute_curvature=config['preprocessing']['compute_curvature'],
        detect_features=config['preprocessing']['detect_features']
    )
    
    # Create data module
    data_module = MeshDataModule(
        data_dir=config['data']['raw_dir'],
        processor=processor,
        batch_size=config['training']['batch_size'],
        train_split=config['data']['train_split'],
        val_split=config['data']['val_split'],
        test_split=config['data']['test_split']
    )
    data_module.setup()
    
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
    
    # Training loop
    best_val_loss = float('inf')
    patience = config['training']['early_stopping_patience']
    patience_counter = 0
    
    for epoch in range(config['training']['num_epochs']):
        # Train
        train_loss = train_epoch(
            model,
            data_module.train_dataloader(),
            optimizer,
            device
        )
        
        # Validate
        val_loss = validate(
            model,
            data_module.val_dataloader(),
            device
        )
        
        # Log progress
        logging.info(
            f'Epoch {epoch + 1}/{config["training"]["num_epochs"]} - '
            f'Train Loss: {train_loss:.4f} - '
            f'Val Loss: {val_loss:.4f}'
        )
        
        # Save checkpoint if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(
                model,
                optimizer,
                epoch + 1,
                val_loss,
                Path(config['training']['checkpoint_dir'])
            )
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            logging.info(f'Early stopping triggered after {epoch + 1} epochs')
            break

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='src/configs/default.yaml')
    parser.add_argument('--checkpoint-dir', type=str, default=None)
    args = parser.parse_args()
    
    main(args.config, args.checkpoint_dir) 