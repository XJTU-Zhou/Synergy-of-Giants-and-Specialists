# -*- coding: utf-8 -*-

"""
src.utils - Utility Package for the Project.

This package provides common, reusable functions that are used across different
modules of the project. It includes:
-   Metrics calculation for end-to-end evaluation (SeqAcc, KPA).
-   Helper functions for loading and processing knowledge graph data.
"""

from .metrics import calculate_sequence_accuracy, calculate_key_parameter_acceptability
from .kg_utils import load_kg_triples, build_entity_relation_maps

# Define what is imported when a user writes "from src.utils import *"
__all__ = [
    'calculate_sequence_accuracy',
    'calculate_key_parameter_acceptability',
    'load_kg_triples',
    'build_entity_relation_maps'
]