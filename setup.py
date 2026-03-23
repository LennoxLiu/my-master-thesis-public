from setuptools import setup, find_packages

# Only including top-level packages to keep the installation flexible
INSTALL_REQUIRES = [
    "numpy>=2.0.0",
    "pandas>=2.0.0",
    "scipy>=1.10.0",
    "scikit-learn>=1.0.0",
    "matplotlib>=3.5.0",
    "seaborn>=0.12.0",
    "torch>=2.0.0",
    "torchvision",
    "torchaudio",
    "tqdm",
    "PyYAML",
    "plotly",
    "optuna",
    "networkx",
    "h5py",
]

setup(
    name="my_thesis_project",
    version="1.0.0",
    author="Lennox Liu",
    description="This project provides a high-performance pipeline for estimating Transfer Entropy (TE) between continuous-time event sequences using deep learning.",
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    python_requires='>=3.8', # Updated to 3.8+ as modern torch/numpy prefer it
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License", # Update if different
        "Operating System :: OS Independent",
    ],
)