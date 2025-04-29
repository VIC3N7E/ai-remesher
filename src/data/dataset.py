import torch
from torch.utils.data import Dataset
import trimesh
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from .mesh_processor import MeshProcessor

class MeshDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        processor: MeshProcessor,
        split: str = 'train',
        transform: Optional[callable] = None
    ):
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.split = split
        self.transform = transform
        
        # Load mesh paths
        self.mesh_paths = self._load_mesh_paths()
        
    def _load_mesh_paths(self) -> List[Path]:
        """Load paths to mesh files."""
        mesh_paths = []
        for ext in ['.obj', '.fbx', '.stl']:
            mesh_paths.extend(list(self.data_dir.glob(f'**/*{ext}')))
        return mesh_paths
    
    def __len__(self) -> int:
        return len(self.mesh_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Load mesh
        mesh_path = self.mesh_paths[idx]
        mesh = trimesh.load(str(mesh_path))
        
        # Process mesh
        data = self.processor.process_mesh(mesh)
        
        # Apply transforms if any
        if self.transform is not None:
            data = self.transform(data)
        
        return data

class MeshDataModule:
    def __init__(
        self,
        data_dir: str,
        processor: MeshProcessor,
        batch_size: int = 32,
        num_workers: int = 4,
        train_split: float = 0.8,
        val_split: float = 0.1,
        test_split: float = 0.1
    ):
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Compute splits
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        
        # Create datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
    def setup(self):
        """Setup datasets."""
        # Load all mesh paths
        all_paths = []
        for ext in ['.obj', '.fbx', '.stl']:
            all_paths.extend(list(self.data_dir.glob(f'**/*{ext}')))
        
        # Shuffle paths
        np.random.shuffle(all_paths)
        
        # Split paths
        n = len(all_paths)
        train_end = int(n * self.train_split)
        val_end = train_end + int(n * self.val_split)
        
        train_paths = all_paths[:train_end]
        val_paths = all_paths[train_end:val_end]
        test_paths = all_paths[val_end:]
        
        # Create datasets
        self.train_dataset = MeshDataset(
            data_dir=self.data_dir,
            processor=self.processor,
            split='train'
        )
        
        self.val_dataset = MeshDataset(
            data_dir=self.data_dir,
            processor=self.processor,
            split='val'
        )
        
        self.test_dataset = MeshDataset(
            data_dir=self.data_dir,
            processor=self.processor,
            split='test'
        )
    
    def train_dataloader(self):
        """Get training dataloader."""
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
    
    def val_dataloader(self):
        """Get validation dataloader."""
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
    
    def test_dataloader(self):
        """Get test dataloader."""
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        ) 