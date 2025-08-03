"""
Setup script for PhoMem-CoSim.
"""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="phomem-cosim",
    version="0.1.0",
    author="Terragon Labs",
    author_email="terry@terragonlabs.com",
    description="Photonic-Memristor Neuromorphic Co-Simulation Platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/terragonlabs/phomem-cosim",
    project_urls={
        "Bug Tracker": "https://github.com/terragonlabs/phomem-cosim/issues",
        "Documentation": "https://phomem-cosim.readthedocs.io/",
        "Source Code": "https://github.com/terragonlabs/phomem-cosim",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "gpu": ["jax[cuda12_pip]"],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "plotting": [
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
        ],
        "hardware": [
            "ngspice-python>=0.1.0",
            "gdspy>=1.6.0",
            "scikit-rf>=0.20.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "phomem-simulate=phomem.cli:simulate",
            "phomem-train=phomem.cli:train",
            "phomem-analyze=phomem.cli:analyze",
        ],
    },
    include_package_data=True,
    package_data={
        "phomem": [
            "data/*.yaml",
            "data/*.json",
            "examples/*.py",
            "tests/*.py",
        ],
    },
    keywords=[
        "photonics",
        "memristors", 
        "neuromorphic",
        "neural networks",
        "simulation",
        "machine learning",
        "hardware",
        "co-simulation",
        "differentiable programming"
    ],
    zip_safe=False,
)