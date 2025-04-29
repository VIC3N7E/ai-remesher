# AI Remesher

A deep learning-based approach for automatic mesh retopology, focusing on producing high-quality, production-ready mesh topology.

## Features

- Automatic mesh retopology with feature preservation
- Quad-dominant mesh generation
- UV seam and normal preservation
- Feature line detection and preservation
- Topology-aware simplification
- Progressive training strategy

## Project Structure

```
ai-remesher/
├── src/
│   ├── models/         # Neural network architectures
│   ├── data/           # Data processing and loading
│   ├── utils/          # Utility functions
│   └── configs/        # Configuration files
├── tests/              # Unit tests
├── examples/           # Example usage and demos
├── data/
│   ├── raw/           # Raw input meshes
│   └── processed/     # Processed training data
└── checkpoints/       # Model checkpoints
```

## Installation

1. Create a new conda environment:
```bash
conda create -n ai-remesher python=3.10
conda activate ai-remesher
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your input meshes in `data/raw/`
2. Run preprocessing:
```bash
python src/data/preprocess.py
```

3. Train the model:
```bash
python src/train.py
```

4. Run inference:
```bash
python src/inference.py --input path/to/mesh.obj --output path/to/output.obj
```

## Development

- Code style: Black + isort
- Type checking: mypy
- Linting: pylint
- Testing: pytest

## License

ISEP License
