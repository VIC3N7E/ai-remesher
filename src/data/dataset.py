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
        transform: Optional[callable] = None,
        supported_formats: List[str] = ['.obj', '.off', '.stl', '.ply', '.fbx']
    ):
        """
        Generic mesh dataset that recursively finds meshes in train/test directories.
        
        Args:
            data_dir (str): Root directory containing the data
            processor (MeshProcessor): Processor for mesh data
            split (str): Either 'train' or 'test'
            transform (callable, optional): Optional transform to be applied on a sample
            supported_formats (list): List of supported mesh file extensions
        """
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.split = split
        self.transform = transform
        self.supported_formats = supported_formats
        
        # Load mesh paths
        self.mesh_paths = self._find_mesh_files()
        if not self.mesh_paths:
            raise ValueError(f"No mesh files found in {split} directories under {data_dir}")
        
        print(f"Found {len(self.mesh_paths)} mesh files for {split} split")
        
    def _find_mesh_files(self) -> List[Path]:
        """
        Recursively find all mesh files in directories named as the split (train/test).
        Returns a list of paths to mesh files.
        """
        mesh_paths = []
        
        # Find all directories named as the split
        split_dirs = list(self.data_dir.rglob(self.split))
        
        for split_dir in split_dirs:
            if not split_dir.is_dir():
                continue
                
            # Find all mesh files in this directory and its subdirectories
            for ext in self.supported_formats:
                mesh_paths.extend(list(split_dir.rglob(f"*{ext}")))
        
        return sorted(mesh_paths)  # Sort for reproducibility
    
    def get_mesh_info(self, mesh_path: Path) -> Dict:
        """
        Extract additional information from mesh path.
        Override this method to extract dataset-specific information.
        """
        return {
            'category': mesh_path.parent.name,
            'filename': mesh_path.name,
            'filepath': str(mesh_path)
        }
    
    def __len__(self) -> int:
        return len(self.mesh_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Load mesh
        mesh_path = self.mesh_paths[idx]
        try:
            mesh = trimesh.load(str(mesh_path))
        except Exception as e:
            print(f"Error loading mesh {mesh_path}: {str(e)}")
            # Return next valid mesh
            return self.__getitem__((idx + 1) % len(self))
        
        # Get mesh information
        mesh_info = self.get_mesh_info(mesh_path)
        
        # Process mesh
        try:
            processed_data = self.processor.process_mesh(mesh)
            data = {
                **processed_data,
                **mesh_info
            }
        except Exception as e:
            print(f"Error processing mesh {mesh_path}: {str(e)}")
            # Return next valid mesh
            return self.__getitem__((idx + 1) % len(self))
        
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
        supported_formats: List[str] = ['.obj', '.off', '.stl', '.ply', '.fbx']
    ):
        """
        Data module for mesh datasets.
        
        Args:
            data_dir (str): Root directory containing the data
            processor (MeshProcessor): Processor for mesh data
            batch_size (int): Batch size for dataloaders
            num_workers (int): Number of workers for dataloaders
            supported_formats (list): List of supported mesh file extensions
        """
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.supported_formats = supported_formats
        
        # Create datasets
        self.train_dataset = None
        self.test_dataset = None
        
    def setup(self):
        """Setup train and test datasets."""
        # Create datasets
        self.train_dataset = MeshDataset(
            data_dir=self.data_dir,
            processor=self.processor,
            split='train',
            supported_formats=self.supported_formats
        )
        
        self.test_dataset = MeshDataset(
            data_dir=self.data_dir,
            processor=self.processor,
            split='test',
            supported_formats=self.supported_formats
        )
        
        print(f"Train dataset size: {len(self.train_dataset)}")
        print(f"Test dataset size: {len(self.test_dataset)}")
    
    def train_dataloader(self):
        """Get training dataloader."""
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
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