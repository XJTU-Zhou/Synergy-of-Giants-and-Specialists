# -*- coding: utf-8 -*-

"""
src - The Core Source Code Package for the "Synergy of Giants and Specialists" Project.

This package contains the three main modules of the framework as described in the paper:
1.  perception_module: Handles the multimodal perception and translation using a fine-tuned LLaVA model.
2.  knowledge_module: Manages knowledge graph representation with the TransEx embedding model.
3.  reasoning_module: Conducts interpretable reasoning and pathfinding using a DRL agent.

"""

import os
import sys

# Add the project's root directory to the Python path.
# This ensures that modules can be imported consistently across different scripts,
# whether they are run from the root directory or from subdirectories.
# For example, it allows scripts in 'scripts/' to import from 'src/' without issues.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# You could optionally provide convenient top-level imports here.
# For instance, if you have a main "pipeline" class that orchestrates everything,
# you could import it here to make it accessible directly from `src`.
#
# from .pipeline import FullPipeline
#
# __all__ = ['FullPipeline']

print("Initializing 'src' package for Synergy of Giants and Specialists project...")