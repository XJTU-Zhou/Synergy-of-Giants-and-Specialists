# -*- coding: utf-8 -*-

"""
src.reasoning_module - The Interpretable Reasoning Module.

This package is the core decision-making component of the framework. It models the
manufacturing process planning task as a reinforcement learning problem on a knowledge graph.

Key Components:
- KGEnvironment: An OpenAI Gym-compliant environment that wraps the manufacturing knowledge graph,
  defining the states, actions, and rewards for the DRL agent.
- Actor/Critic: PyTorch neural network models that form the policy and value functions
  for the GAE-PPO agent.

This module takes the structured entities from the perception module and the KG embeddings
from the knowledge module to perform autonomous, multi-hop pathfinding.
"""

from .environment import KGEnvironment
from .agent import Actor, Critic

__all__ = [
    'KGEnvironment',
    'Actor',
    'Critic'
]

print("Initializing 'reasoning_module' package...")