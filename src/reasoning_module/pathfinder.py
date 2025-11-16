# -*- coding: utf-8 -*-

import torch
import numpy as np
from typing import List, Tuple, Dict

from src.reasoning_module.agent import Actor, Critic
from src.reasoning_module.environment import KGEnvironment
from tianshou.policy import PPOPolicy

class PathFinder:
    """
    Uses a trained DRL agent to find interpretable paths in the knowledge graph.
    """

    def __init__(self, config: Dict, kg: List, embeddings: np.ndarray, entity_vocab: Dict, relation_vocab: Dict):
        """
        Initializes the PathFinder.

        Args:
            config (Dict): The DRL configuration dictionary.
            kg (List): The knowledge graph data.
            embeddings (np.ndarray): Pre-trained entity embeddings.
            entity_vocab (Dict): Mapping from entity name to ID.
            relation_vocab (Dict): Mapping from relation name to ID.
        """
        self.config = config
        self.kg = kg
        self.embeddings = embeddings
        self.entity_vocab = entity_vocab
        self.relation_vocab = relation_vocab
        self.id_to_entity = {v: k for k, v in entity_vocab.items()}
        self.id_to_relation = {v: k for k, v in relation_vocab.items()}
        
        self.device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
        
        # Initialize the environment for inference
        self.env = KGEnvironment(kg=self.kg, embeddings=self.embeddings, max_hops=self.config['env']['max_hops'])
        
        # Load the trained policy
        self.policy = self._load_policy()

    def _load_policy(self) -> PPOPolicy:
        """Loads the pre-trained PPO policy from disk."""
        state_shape = self.env.observation_space.shape[0]
        action_shape = self.env.action_space.n
        
        actor_net = Actor(state_dim=state_shape, action_dim=action_shape, hidden_dim=self.config['model']['hidden_dims'][0]).to(self.device)
        critic_net = Critic(state_dim=state_shape, hidden_dim=self.config['model']['hidden_dims'][0]).to(self.device)
        optim = torch.optim.Adam(
            list(actor_net.parameters()) + list(critic_net.parameters()), 
            lr=self.config['training']['learning_rate']
        )
        
        policy = PPOPolicy(
            actor=actor_net, critic=critic_net, optim=optim,
            dist_fn=torch.distributions.Categorical,
            action_space=self.env.action_space
        )
        
        policy_path = self.config['model']['save_path']
        if not os.path.exists(policy_path):
            raise FileNotFoundError(f"Trained policy not found at {policy_path}. Please train the agent first.")
        
        policy.load_state_dict(torch.load(policy_path, map_location=self.device))
        policy.eval() # Set the policy to evaluation mode
        print(f"Policy loaded successfully from {policy_path}")
        return policy

    def find_path(self, start_entity_name: str, target_entity_name: str) -> Tuple[List[str], List[str]]:
        """
        Finds the most likely path between two entities using the trained agent.

        Args:
            start_entity_name (str): The name of the starting entity.
            target_entity_name (str): The name of the target entity.

        Returns:
            A tuple containing:
            - path_entities (List[str]): List of entity names in the found path.
            - path_relations (List[str]): List of relation names connecting the entities.
        """
        if start_entity_name not in self.entity_vocab or target_entity_name not in self.entity_vocab:
            print("Error: Start or target entity not in vocabulary.")
            return [], []

        start_id = self.entity_vocab[start_entity_name]
        target_id = self.entity_vocab[target_entity_name]
        
        obs = self.env.reset(start_entity=start_id, target_entity=target_id)
        done = False
        
        path_relations = []

        while not done:
            # Prepare observation for the policy
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # --- Action Masking during Inference ---
            current_entity = self.env.state['current_entity']
            possible_actions = self.env.adj.get(current_entity, [])
            action_mask = torch.zeros(self.env.action_space.n, dtype=torch.bool)
            if possible_actions:
                action_mask[:len(possible_actions)] = 1
            
            # Get action from the policy (deterministic for inference)
            with torch.no_grad():
                dist = self.policy.actor(obs_tensor, action_mask=action_mask)
                action = dist.probs.argmax().item() # Choose the most likely action

            # Step the environment
            obs, reward, done, info = self.env.step(action)
            
            # Record the chosen relation
            if "status" not in ["invalid_action_or_dead_end"]:
                 chosen_relation_id, _ = possible_actions[action]
                 path_relations.append(self.id_to_relation[chosen_relation_id])
            
            if done:
                print(f"Pathfinding finished. Status: {info.get('status')}")
                break

        path_entities = [self.id_to_entity[eid] for eid in self.env.state['path']]

        return path_entities, path_relations


if __name__ == '__main__':
    # Example usage of the PathFinder
    # This demonstrates how the end-to-end evaluation script would use this class
    
    # 1. Load config and data (mocked for this example)
    from src.utils.kg_utils import load_kg_and_embeddings
    
    config_file = 'configs/stage3_drl_config.yaml'
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    kg, embeddings, entity_vocab, relation_vocab = load_kg_and_embeddings(
        config['data']['kg_embedding_dataset_path'],
        config['data']['embeddings_path']
    )
    
    # 2. Initialize PathFinder
    finder = PathFinder(config, kg, embeddings, entity_vocab, relation_vocab)

    # 3. Find a path
    # Replace with actual entities from your KG for a real test
    start_node = 'gear_shaft' 
    target_node = 'external_grinding' 
    
    if start_node in entity_vocab and target_node in entity_vocab:
        entities, relations = finder.find_path(start_node, target_node)
        
        print("\n--- Found Path ---")
        path_str = entities[0]
        for i, rel in enumerate(relations):
            path_str += f" --[{rel}]--> {entities[i+1]}"
        print(path_str)
        print("------------------")
    else:
        print(f"Could not run example: '{start_node}' or '{target_node}' not in KG.")