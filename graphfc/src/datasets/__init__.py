"""
Datasets module for GraphFC framework.
"""

from .loaders import (
    FactCheckingExample,
    HOVERDatasetLoader,
    FEVEROUSDatasetLoader, 
    SciFacitDatasetLoader,
    load_dataset,
    get_dataset_info
)

__all__ = [
    "FactCheckingExample",
    "HOVERDatasetLoader",
    "FEVEROUSDatasetLoader",
    "SciFacitDatasetLoader", 
    "load_dataset",
    "get_dataset_info"
]