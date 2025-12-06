import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from typing import Tuple, List
from app.agents.base_agent import BaseAgent
from app.environment.tennis_env import TennisEnv, Action
from app.models.env import State


class DQNNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent(BaseAgent):
    def __init__(
        self,
        env: TennisEnv,
        lr: float = 0.001,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        memory_size: int = 10000,
        batch_size: int = 32,
        target_update_freq: int = 100
    ):
        self.env = env
        self.state_size = len(env.state)
        self.action_size = len(env.action_space)
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Experience replay buffer
        self.memory = deque(maxlen=memory_size)
        
        # Neural networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DQNNetwork(self.state_size, self.action_size).to(self.device)
        self.target_network = DQNNetwork(self.state_size, self.action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Update target network
        self.update_target_network()
        
        self.step_count = 0
    
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state: State) -> Action:
        """Choose action using epsilon-greedy policy"""
        if random.random() <= self.epsilon:
            # Random action
            return self._random_action()
        
        # Use neural network to choose action
        state_tensor = torch.FloatTensor(state.encode(self.env)).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        action_idx = torch.argmax(q_values).item()
        
        return self._idx_to_action(action_idx)
    
    def _random_action(self) -> Action:
        """Generate random valid action"""
        # Implement based on your Action class
        # This is a placeholder - adjust based on your action space
        action = random.choice(list(self.env.action_space))
        
        return Action(
            shot_type=action[0],
            shot_direction=action[1]
        )
    
    def _idx_to_action(self, idx: int) -> Action:
        """Convert action index to Action object"""
        # Implement based on your action space encoding
        action = list(self.env.action_space)[idx]
        
        return Action(
            shot_type=action[0],
            shot_direction=action[1]
        )
    
    def replay(self):
        """Train the model on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return None
        
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0].encode(self.env) for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3].encode(self.env) for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.update_target_network()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()  # Return loss value for logging
    
    def save(self, filepath: str):
        """Save model weights"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count
        }, filepath)
    
    def load(self, filepath: str):
        """Load model weights"""
        checkpoint = torch.load(filepath)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.step_count = checkpoint['step_count']