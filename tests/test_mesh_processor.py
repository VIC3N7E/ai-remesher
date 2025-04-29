import pytest
import numpy as np
import trimesh
from src.data.mesh_processor import MeshProcessor

@pytest.fixture
def sample_mesh():
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
    return trimesh.Trimesh(vertices=vertices, faces=faces)

@pytest.fixture
def mesh_processor():
    config = {
        'preprocessing': {
            'angle_threshold': 30,
            'normalize': True
        }
    }
    return MeshProcessor(config)

def test_mesh_normalization(mesh_processor, sample_mesh):
    normalized_mesh, transform_params = mesh_processor.normalize_mesh(sample_mesh)
    
    # Check if mesh is centered
    assert np.allclose(normalized_mesh.vertices.mean(axis=0), [0, 0, 0], atol=1e-6)
    
    # Check if mesh is scaled to unit sphere
    assert np.max(np.linalg.norm(normalized_mesh.vertices, axis=1)) <= 1.0 + 1e-6
    
    # Check if transform parameters are stored
    assert 'centroid' in transform_params
    assert 'scale' in transform_params

def test_curvature_computation(mesh_processor, sample_mesh):
    curvatures = mesh_processor.compute_curvature(sample_mesh)
    
    # Check if curvatures are computed for all vertices
    assert len(curvatures) == len(sample_mesh.vertices)
    
    # Check if curvatures are within reasonable range
    assert np.all(np.isfinite(curvatures))

def test_feature_line_detection(mesh_processor, sample_mesh):
    feature_edges = mesh_processor.detect_feature_lines(sample_mesh)
    
    # Check if feature edges are detected
    assert isinstance(feature_edges, np.ndarray)
    assert feature_edges.shape[1] == 2  # Each edge has 2 vertices 