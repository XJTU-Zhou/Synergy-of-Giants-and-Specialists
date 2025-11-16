# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class Actor(nn.Module):
    """
    The Actor (Policy) Network.
    
    Takes a state representation (entity embedding) and outputs a probability
    distribution over the possible actions (relations to follow).
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        """
        Initializes the Actor network.

        Args:
            state_dim (int): The dimension of the input state (embedding dimension).
            action_dim (int): The maximum number of possible actions.
            hidden_dim (int): The size of the hidden layers.
        """
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(state_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor, action_mask: torch.Tensor = None) -> Categorical:
        """
        Performs a forward pass to get the action distribution.

        Args:
            state (torch.Tensor): The current state tensor.
            action_mask (torch.Tensor, optional): A binary mask to filter invalid actions.
                                                  A value of 1 indicates a valid action. Defaults to None.

        Returns:
            torch.distributions.Categorical: A categorical distribution over valid actions.
        """
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        action_logits = self.layer3(x)

        if action_mask is not None:
            # Apply the mask: set logits of invalid actions to a very small number
            action_logits[action_mask == 0] = -1e9
        
        # Use softmax to get probabilities
        action_probs = F.softmax(action_logits, dim=-1)
        
        # Create a categorical distribution for sampling actions
        dist = Categorical(action_probs)
        
        return dist


class Critic(nn.Module):
    """
    The Critic (Value) Network.

    Takes a state representation (entity embedding) and outputs a single scalar value
    representing the expected cumulative reward from that state (V(s)).
    """
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        """
        Initializes the Critic network.

        Args:
            state_dim (int): The dimension of the input state (embedding dimension).
            hidden_dim (int): The size of the hidden layers.
        """
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(state_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, 1) # Output a single value

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass to estimate the state value.

        Args:
            state (torch.Tensor): The current state tensor.

        Returns:
            torch.Tensor: The estimated value of the state.
        """
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        state_value = self.layer3(x)
        return state_value