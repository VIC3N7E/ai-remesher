import pytest
import torch
import os
import tempfile
import numpy as np
import trimesh
from src.data.dataset import MeshDataset

@pytest.fixture
def sample_mesh_file():
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a simple cube mesh
        vertices = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
        ])
        faces = np.array([
            [0, 1, 2], [0, 2, 3],  # bottom
            [4, 5, 6], [4, 6, 7],  # top
            [0, 1, 5], [0, 5, 4],  # front
            [2, 3, 7], [2, 7, 6],  # back
            [0, 3, 7], [0, 7, 4],  # left
            [1, 2, 6], [1, 6, 5]   # right
        ])
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        # Save mesh to file
        mesh_path = os.path.join(tmpdir, 'cube.obj')
        mesh.export(mesh_path)
        yield mesh_path

@pytest.fixture
def dataset_config():
    return {
        'preprocessing': {
            'angle_threshold': 30,
            'normalize': True
        }
    }

def test_dataset_initialization(tmp_path, dataset_config):
    # Create dataset with temporary directory
    dataset = MeshDataset(
        root=str(tmp_path),
        config=dataset_config
    )
    
    assert isinstance(dataset, MeshDataset)
    assert dataset.config == dataset_config

def test_dataset_processing(tmp_path, sample_mesh_file, dataset_config):
    # Create raw directory and copy sample mesh
    raw_dir = os.path.join(tmp_path, 'raw')
    os.makedirs(raw_dir, exist_ok=True)
    os.system(f'cp {sample_mesh_file} {raw_dir}/')
    
    # Create dataset
    dataset = MeshDataset(
        root=str(tmp_path),
        config=dataset_config
    )
    
    # Process dataset
    dataset.process()
    
    # Check if processed files exist
    processed_dir = os.path.join(tmp_path, 'processed')
    assert os.path.exists(processed_dir)
    assert len(os.listdir(processed_dir)) > 0

def test_dataset_get_item(tmp_path, sample_mesh_file, dataset_config):
    # Create raw directory and copy sample mesh
    raw_dir = os.path.join(tmp_path, 'raw')
    os.makedirs(raw_dir, exist_ok=True)
    os.system(f'cp {sample_mesh_file} {raw_dir}/')
    
    # Create and process dataset
    dataset = MeshDataset(
        root=str(tmp_path),
        config=dataset_config
    )
    dataset.process()
    
    # Get first item
    data = dataset[0]
    
    # Check data structure
    assert hasattr(data, 'x')
    assert hasattr(data, 'pos')
    assert hasattr(data, 'normal')
    assert hasattr(data, 'curvature')
    assert hasattr(data, 'edge_index')
    assert hasattr(data, 'edge_attr')
    assert hasattr(data, 'face')
    
    # Check data types
    assert isinstance(data.x, torch.Tensor)
    assert isinstance(data.pos, torch.Tensor)
    assert isinstance(data.normal, torch.Tensor)
    assert isinstance(data.curvature, torch.Tensor)
    assert isinstance(data.edge_index, torch.Tensor)
    assert isinstance(data.edge_attr, torch.Tensor)
    assert isinstance(data.face, torch.Tensor) 