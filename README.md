# AI Remesher

A mesh processing pipeline for feature extraction, visualization, and retopology.

## Features

- Mesh feature extraction (curvature, edges, corners, creases)
- Interactive 3D visualization using PyVista
- Mesh export in various formats (OBJ, PLY, STL)
- Configurable processing pipeline

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai-remesher.git
cd ai-remesher
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install the package in development mode:
```bash
pip install -e .
```

## Testing

To test the mesh processing pipeline:

1. Create sample meshes for testing (optional):
```bash
python create_sample_mesh.py
```
This will create sample meshes in the `sample_meshes` directory.

2. Run the test script:
```bash
python run_test.py --mesh path/to/your/mesh.obj
```

Or if you created sample meshes:
```bash
python run_test.py --mesh sample_meshes/sphere.obj
```

Optional arguments:
- `--config`: Path to configuration file (default: configs/default.yaml)
- `--output`: Output directory for results (default: output/test)

The test will:
- Load and process the mesh
- Extract features (curvature, edges, corners, creases)
- Generate visualizations
- Export the processed mesh with features

Results will be saved in the output directory:
- `mesh.png`: Original mesh visualization
- `features.png`: Feature visualization
- `curvature.png`: Curvature visualization
- `feature_edges.png`: Feature edges visualization
- `feature_corners.png`: Feature corners visualization
- `feature_creases.png`: Feature creases visualization
- `mesh_with_features.obj`: Exported mesh with features

## Configuration

The pipeline can be configured through the YAML configuration file. See `configs/default.yaml` for available options.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{ai_remesher,
  title = {AI Remesher},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/ai-remesher}
}
```
