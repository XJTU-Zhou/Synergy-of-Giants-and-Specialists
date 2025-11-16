# -*- coding: utf-8 -*-

"""
src.knowledge_module - Stage 2: Knowledge Representation Module.

This module is responsible for building a rich, continuous semantic "map" from
the discrete manufacturing Knowledge Graph (KG).

It contains:
-   dataset.py: Utilities for loading KG triples and creating PyTorch DataLoaders.
-   model.py: The implementation of the TransEx model, a Transformer-based
    KG embedding model that uses complex-valued embeddings to capture complex
    and asymmetric relations.
"""

from .dataset import KGDataset, create_dataloaders
from .model import TransEx

__all__ = [
    'KGDataset',
    'create_dataloaders',
    'TransEx'
]