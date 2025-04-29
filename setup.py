#!/usr/bin/env python3
"""
Setup script for the AI Remesher package.
"""

from setuptools import setup, find_packages

setup(
    name="ai-remesher",
    version="0.1.0",
    description="AI-powered mesh remeshing tool",
    author="GECAD",
    author_email="info@gecad.isep.ipp.pt",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "trimesh>=3.9.0",
        "pyvista>=0.34.0",
        "matplotlib>=3.5.0",
        "scipy>=1.7.0",
        "torch>=1.10.0",
        "torchvision>=0.11.0",
        "tqdm>=4.62.0",
        "pyyaml>=6.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: 3D Modeling",
    ],
) 