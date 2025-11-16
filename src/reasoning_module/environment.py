# -*- coding: utf-8 -*-

import gym
import numpy as np
from gym import spaces
from typing import List, Tuple, Dict, Any

class KGEnvironment(gym.Env):
    """
    A reinforcement learning environment for navigating a manufacturing Knowledge Graph.

    This environment models the process planning task as a pathfinding problem where an
    agent starts at an initial entity (e.g., a part) and must find a path to a
    target entity (e.g., a required parameter or process).

    Attributes:
        kg (List[Tuple[int, int, int]]): The knowledge graph represented as triples of entity/relation IDs.
        embeddings (np.ndarray): Pre-trained TransEx embeddings for all entities.
        entity_vocab (Dict[str, int]): Mapping from entity name to its ID.
        relation_vocab (Dict[str, int]): Mapping from relation name to its ID.
        max_hops (int): Maximum number of steps per episode.
        
        state (Dict[str, Any]): The current state of the environment.
        action_space (spaces.Discrete): The action space, sized to the max number of relations.
        observation_space (spaces.Box): The observation space, defined by the embedding dimension.
    """

    def __init__(self, kg: List[Tuple[int, int, int]], embeddings: np.ndarray, max_hops: int):
        """
        Initializes the Knowledge Graph Environment.

        Args:
            kg (List[Tuple[int, int, int]]): List of KG triples (head_id, relation_id, tail_id).
            embeddings (np.ndarray): A NumPy array of shape (num_entities, embedding_dim).
            max_hops (int): The maximum allowed path length for an episode.
        """
        super(KGEnvironment, self).__init__()

        self.kg = kg
        self.embeddings = embeddings
        self.max_hops = max_hops
        
        self.embedding_dim = embeddings.shape[1]

        # Pre-process KG for efficient lookups
        self.adj = self._build_adjacency_list()
        
        # Define action and observation spaces
        # The action space is dynamic, but for Gym compatibility, we set it to the total
        # number of possible relations. Action masking will be handled by the agent.
        self.num_relations = len(set(r for _, r, _ in kg))
        self.action_space = spaces.Discrete(self.num_relations * 2) # Relations can be forward or backward

        # Observation is the embedding of the current entity
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.embedding_dim,), dtype=np.float32
        )

        self.state = {}

    def _build_adjacency_list(self) -> Dict[int, List[Tuple[int, int]]]:
        """Builds an adjacency list for efficient action sampling."""
        adj = {i: [] for i in range(len(self.embeddings))}
        for h, r, t in self.kg:
            adj[h].append((r, t))
        return adj

    def _get_observation(self) -> np.ndarray:
        """Returns the embedding of the current entity as the observation."""
        current_entity_id = self.state['current_entity']
        return self.embeddings[current_entity_id]

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculates cosine similarity between two vectors."""
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 0.0
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def reset(self, start_entity: int, target_entity: int) -> np.ndarray:
        """
        Resets the environment to a new starting state for a new episode.

        Args:
            start_entity (int): The ID of the starting entity for the pathfinding task.
            target_entity (int): The ID of the target entity.

        Returns:
            np.ndarray: The initial observation (embedding of the start entity).
        """
        self.state = {
            'start_entity': start_entity,
            'target_entity': target_entity,
            'current_entity': start_entity,
            'path': [start_entity],
            'current_step': 0
        }
        return self._get_observation()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Executes one step in the environment.

        Args:
            action (int): The ID of the chosen action (relation_id, next_entity_id).

        Returns:
            Tuple containing:
            - observation (np.ndarray): The new observation.
            - reward (float): The reward for the taken action.
            - done (bool): Whether the episode has ended.
            - info (Dict[str, Any]): Auxiliary diagnostic information.
        """
        if self.state['current_step'] >= self.max_hops:
            # Should not happen if handled correctly, but as a safeguard
            return self._get_observation(), 0.0, True, {"status": "max_hops_exceeded"}

        # Get all possible actions from the current state
        possible_actions = self.adj.get(self.state['current_entity'], [])
        
        if not possible_actions or action >= len(possible_actions):
            # Invalid action chosen (e.g., due to no masking) or dead end
            done = True
            reward = -1.0 # Penalize for invalid moves
            info = {"status": "invalid_action_or_dead_end"}
            return self._get_observation(), reward, done, info

        # Execute the chosen valid action
        chosen_relation, next_entity = possible_actions[action]
        
        self.state['current_entity'] = next_entity
        self.state['path'].append(next_entity)
        self.state['current_step'] += 1

        # Calculate reward based on paper's formula (Eq. 17)
        target_entity = self.state['target_entity']
        if next_entity == target_entity:
            reward = 1.0
            done = True
            info = {"status": "target_reached"}
        else:
            # Dense reward: semantic similarity to the target
            current_embedding = self.embeddings[next_entity]
            target_embedding = self.embeddings[target_entity]
            reward = 0.5 * self._cosine_similarity(current_embedding, target_embedding)
            done = self.state['current_step'] >= self.max_hops
            info = {"status": "in_progress"}
        
        if done and info.get("status") != "target_reached":
            info["status"] = "max_hops_reached"

        return self._get_observation(), reward, done, info

    def render(self, mode='human'):
        """Renders the current state of the environment (optional)."""
        print(f"Step: {self.state['current_step']}, Path: {self.state['path']}")