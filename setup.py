from setuptools import setup, find_packages

setup(
    name="star-st",
    version="0.1.0",
    description="StaR: Stability-Aware Representation Learning for Spatial Transcriptomics",
    author="Ziheng Duan",
    url="https://github.com/RRRussell/StaR",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "torch-geometric>=2.1.0",
        "scanpy>=1.9.0",
        "anndata>=0.8.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "tqdm>=4.62.0",
    ],
    extras_require={
        "mclust": ["rpy2>=3.4.0"],
        "dev": ["jupyter", "pytest"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
