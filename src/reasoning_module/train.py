# -*- coding: utf-8 -*-

import os
import yaml
import torch
import numpy as np
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import SubprocVectorEnv
from tianshou.policy import PPOPolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.utils.net.common import Net
from tianshou.utils.net.discrete import Actor as TianshouActor, Critic as TianshouCritic

from src.reasoning_module.environment import KGEnvironment
from src.reasoning_module.agent import Actor, Critic
from src.utils.kg_utils import load_kg_and_embeddings # Assume this util function exists

def train_drl_agent(config_path: str):
    """
    Main function to train the DRL agent for knowledge graph pathfinding.

    Args:
        config_path (str): Path to the DRL configuration YAML file.
    """
    # 1. Load Configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 2. Load KG and Embeddings
    print("Loading knowledge graph and embeddings...")
    kg_data_path = config['data']['kg_embedding_dataset_path']
    embeddings_path = config['data']['embeddings_path']
    kg, embeddings, entity_vocab, relation_vocab = load_kg_and_embeddings(kg_data_path, embeddings_path)
    
    # In a real scenario, you'd load training pairs (start_entity, target_entity)
    # For this example, we'll create some dummy pairs.
    # Replace this with actual training data loading.
    train_tasks = np.random.randint(0, len(entity_vocab), size=(1000, 2)) # 1000 dummy tasks

    # 3. Setup Environment
    print("Setting up environments...")
    def make_env():
        return KGEnvironment(kg=kg, embeddings=embeddings, max_hops=config['env']['max_hops'])

    # Parallel environments for faster training
    train_envs = SubprocVectorEnv([make_env for _ in range(config['training']['num_train_envs'])])
    test_envs = SubprocVectorEnv([make_env for _ in range(config['training']['num_test_envs'])])

    # 4. Setup Agent Networks
    state_shape = embeddings.shape[1]
    # Tianshou requires a specific action_shape, even if it's dynamic in our env
    # The actual valid action space will be handled by masking during policy forward pass
    action_shape = train_envs.action_space[0].n 

    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    
    # Use our custom Actor and Critic models
    net = Net(state_shape, hidden_sizes=config['model']['hidden_dims'], device=device)
    actor_net = Actor(state_dim=state_shape, action_dim=action_shape, hidden_dim=config['model']['hidden_dims'][0]).to(device)
    critic_net = Critic(state_dim=state_shape, hidden_dim=config['model']['hidden_dims'][0]).to(device)
    
    # Optimizer
    optim = torch.optim.Adam(
        list(actor_net.parameters()) + list(critic_net.parameters()), 
        lr=config['training']['learning_rate']
    )

    # 5. Setup PPO Policy
    policy = PPOPolicy(
        actor=actor_net,
        critic=critic_net,
        optim=optim,
        dist_fn=torch.distributions.Categorical,
        # GAE parameters from the paper
        gae_lambda=config['ppo']['gae_lambda'],
        # PPO parameters
        eps_clip=config['ppo']['eps_clip'],
        vf_coef=config['ppo']['vf_coef'],
        ent_coef=config['ppo']['ent_coef'],
        max_grad_norm=config['ppo']['max_grad_norm'],
        reward_normalization=True,
        action_space=train_envs.action_space[0],
        action_scaling=False
    )
    
    # This is a hook to pass the current observation to the actor for action masking
    # Tianshou's PPO doesn't natively support dynamic action masks easily in its standard loop.
    # A more advanced implementation might require subclassing PPOPolicy.
    # For simplicity, we assume the environment handles invalid actions gracefully (with a penalty).
    # A proper masking implementation would look like this:
    # def forward_with_mask(self, batch, state=None, **kwargs):
    #     logits, h = self.actor(batch.obs, state=state, info=batch.info)
    #     if hasattr(batch.info, "action_mask"):
    #         logits[~batch.info.action_mask] = -1e9 # Apply mask
    #     return logits, h
    # policy.actor.forward = forward_with_mask
    

    # 6. Setup Collector
    train_collector = Collector(
        policy, 
        train_envs, 
        VectorReplayBuffer(config['training']['buffer_size'], len(train_envs))
    )
    test_collector = Collector(policy, test_envs)

    # 7. Define Training Logic
    def stop_fn(mean_rewards):
        # Stop training if the average reward reaches a certain threshold
        return mean_rewards >= config['training']['reward_threshold']

    def train_fn(epoch, env_step):
        # Custom logic to set the start/target entities for each episode
        # This is crucial for training on a variety of tasks
        task_indices = np.random.choice(len(train_tasks), size=len(train_envs))
        tasks_to_run = train_tasks[task_indices]
        # Tianshou's SubprocVectorEnv can take a list of reset parameters
        train_envs.reset(start_entity=[t[0] for t in tasks_to_run], target_entity=[t[1] for t in tasks_to_run])
        
    def test_fn(epoch, env_step):
        # Similar logic for testing
        task_indices = np.random.choice(len(train_tasks), size=len(test_envs))
        tasks_to_run = train_tasks[task_indices]
        test_envs.reset(start_entity=[t[0] for t in tasks_to_run], target_entity=[t[1] for t in tasks_to_run])


    # 8. Run Trainer
    print("Starting DRL agent training...")
    result = onpolicy_trainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=config['training']['max_epoch'],
        step_per_epoch=config['training']['step_per_epoch'],
        step_per_collect=config['training']['step_per_collect'],
        repeat_per_collect=config['training']['repeat_per_collect'],
        episode_per_test=config['training']['episode_per_test'],
        batch_size=config['training']['batch_size'],
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=stop_fn,
        show_progress=True
    )

    print(f"Training finished. Final results:\n{result}")

    # 9. Save the trained policy
    save_path = config['model']['save_path']
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(policy.state_dict(), save_path)
    print(f"Policy saved to {save_path}")


if __name__ == '__main__':
    # Example of how to run the training
    # This would typically be called from a script in the `scripts/` directory
    config_file = 'configs/stage3_drl_config.yaml'
    train_drl_agent(config_file)