import trimesh
import numpy as np
from typing import Tuple, Optional
from skimage.measure import marching_cubes

def voxel_remeshing(mesh: trimesh.Trimesh,
                   resolution: float = 0.1,
                   adaptive: bool = True,
                   curvature_threshold: float = 0.1) -> trimesh.Trimesh:
    return mesh