import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import List

from app.agents.base_agent import BaseAgent
from app.environment.tennis_env import TennisEnv, Action
from app.models.env import State


class PolicyNetwork(nn.Module):
    """
    Neural network for the REINFORCE agent.
    It outputs a probability distribution over actions.
    """

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        # Output logits, which will be converted to probabilities
        x = self.fc3(x)
        return torch.softmax(x, dim=-1)


class ReinforceAgent(BaseAgent):
    """
    REINFORCE (Monte Carlo Policy Gradient) Agent.

    This agent learns by collecting trajectories of (state, action, reward)
    for a full episode and then updating its policy based on the discounted
    returns calculated from that episode.
    """

    def __init__(
        self,
        env: TennisEnv,
        lr: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 0.0,
        epsilon_min: float = 0.0,
        epsilon_decay: float = 0.99,
    ):
        self.env = env
        self.state_size = len(env.state)
        self.action_size = len(env.action_space)
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Policy network and optimizer
        self.policy_network = PolicyNetwork(self.state_size, self.action_size).to(
            self.device
        )
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.lr)

        # Episode buffer to store rewards and log probabilities
        self.rewards: List[float] = []
        self.log_probs: List[torch.Tensor] = []

        # Action mapping for quick lookup
        self._action_to_idx_map = {
            action: i for i, action in enumerate(self.env.action_space)
        }
        self._idx_to_action_map = {
            i: action for i, action in enumerate(self.env.action_space)
        }

    def act(self, state: State, training: bool = True) -> Action:
        """
        Selects an action by sampling from the policy distribution.
        """
        state_encoded = state.encode(self.env)
        state_tensor = torch.FloatTensor(state_encoded).to(self.device)

        # Get action probabilities from the policy network
        probs = self.policy_network(state_tensor)

        # Create a categorical distribution and sample an action
        m = Categorical(probs)
        action_idx = m.sample()

        # Store the log probability of the chosen action for the update step
        self.log_probs.append(m.log_prob(action_idx))

        shot_type, shot_direction = self._idx_to_action_map[action_idx.item()]
        return Action(shot_type=shot_type, shot_direction=shot_direction)

    def remember(self, reward):
        """
        Stores the reward for the current step.
        Unlike DQN, it doesn't need the full (s,a,r,s',d) tuple in the same way.
        """
        self.rewards.append(reward)

    def update(self):
        """
        Updates the policy network at the end of an episode using the REINFORCE algorithm.
        """
        discounted_returns = []
        R = 0
        # Calculate discounted returns from the end of the episode to the beginning
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            discounted_returns.insert(0, R)

        # Normalize returns for more stable training
        returns = torch.tensor(discounted_returns).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        policy_loss = []
        for log_prob, R in zip(self.log_probs, returns):
            # The core of REINFORCE: scale log probability by the discounted return
            policy_loss.append(-log_prob * R)

        # Sum the losses, backpropagate, and update the network
        self.optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        self.optimizer.step()

        # Clear the episode buffers for the next episode
        self.rewards.clear()
        self.log_probs.clear()

        return loss.item()

    def save(self, filepath: str):
        """Saves the policy network's state."""
        torch.save(self.policy_network.state_dict(), filepath)

    def load(self, filepath: str):
        """Loads the policy network's state."""
        self.policy_network.load_state_dict(
            torch.load(filepath, map_location=self.device)
        )
        self.policy_network.to(self.device)
