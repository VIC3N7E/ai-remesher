import pytest
import torch
from src.models.retopology_model import RetopologyModel, MeshFeatureExtractor, MeshTransformer

@pytest.fixture
def model_config():
    return {
        'in_channels': 9,
        'hidden_channels': 256,
        'num_heads': 8
    }

@pytest.fixture
def sample_batch():
    # Create a sample batch with 2 graphs, each with 4 vertices
    batch = {
        'x': torch.randn(8, 9),  # 8 vertices (4 per graph) with 9 features each
        'edge_index': torch.tensor([
            [0, 1, 1, 2, 2, 3, 3, 0,  # first graph
             4, 5, 5, 6, 6, 7, 7, 4], # second graph
            [1, 0, 2, 1, 3, 2, 0, 3,  # first graph
             5, 4, 6, 5, 7, 6, 4, 7]  # second graph
        ]),
        'batch': torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])  # vertex to graph assignment
    }
    return batch

def test_feature_extractor(model_config):
    extractor = MeshFeatureExtractor(
        model_config['in_channels'],
        model_config['hidden_channels']
    )
    
    # Test with random input
    x = torch.randn(4, model_config['in_channels'])
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    
    output = extractor(x, edge_index)
    
    # Check output shape
    assert output.shape == (4, model_config['hidden_channels'])
    assert not torch.isnan(output).any()

def test_transformer(model_config):
    transformer = MeshTransformer(
        model_config['hidden_channels'],
        model_config['num_heads']
    )
    
    # Test with random input
    x = torch.randn(4, 1, model_config['hidden_channels'])
    output = transformer(x)
    
    # Check output shape
    assert output.shape == (4, 1, model_config['hidden_channels'])
    assert not torch.isnan(output).any()

def test_full_model(model_config, sample_batch):
    model = RetopologyModel(model_config)
    
    # Test forward pass
    output = model(sample_batch)
    
    # Check output structure
    assert 'vertices' in output
    assert 'normals' in output
    assert 'uvs' in output
    
    # Check output shapes
    assert output['vertices'].shape == (8, 3)  # 8 vertices, 3 coordinates each
    assert output['normals'].shape == (8, 3)   # 8 vertices, 3 normal components each
    assert output['uvs'].shape == (8, 2)       # 8 vertices, 2 UV coordinates each
    
    # Check if normals are normalized
    normal_norms = torch.norm(output['normals'], dim=1)
    assert torch.allclose(normal_norms, torch.ones_like(normal_norms), atol=1e-6) 