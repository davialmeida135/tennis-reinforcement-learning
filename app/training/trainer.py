import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict
from app.environment.tennis_env import TennisEnv, Action, Turn
from app.agents.dqn_agent import DQNAgent


class Trainer:
    def __init__(self, env: TennisEnv, agent: DQNAgent):
        self.env = env
        self.agent = agent
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'win_rates': [],
            'epsilon_values': []
        }
    
    def train(self, episodes: int = 1000, save_freq: int = 100):
        """Train the agent for specified number of episodes"""
        
        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0
            steps = 0
            
            while True:
                # Agent chooses action
                action = self.agent.act(state)
                
                # Environment processes action
                next_state, reward, done, info = self.env.step(action)
                
                # Store experience
                action_idx = self._action_to_idx(action)
                self.agent.remember(state, action_idx, reward, next_state, done)
                
                # Train the agent
                if len(self.agent.memory) > self.agent.batch_size:
                    self.agent.replay()
                
                state = next_state
                total_reward += reward
                steps += 1
                
                if done:
                    break
            
            # Record training metrics
            self.training_history['episode_rewards'].append(total_reward)
            self.training_history['episode_lengths'].append(steps)
            self.training_history['epsilon_values'].append(self.agent.epsilon)
            
            # Calculate win rate (last 100 episodes)
            if episode >= 99:
                recent_rewards = self.training_history['episode_rewards'][-100:]
                win_rate = sum(1 for r in recent_rewards if r > 0) / len(recent_rewards)
                self.training_history['win_rates'].append(win_rate)
            
            # Print progress
            if episode % 100 == 0:
                avg_reward = np.mean(self.training_history['episode_rewards'][-100:])
                print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Epsilon: {self.agent.epsilon:.3f}")
            
            # Save model
            if episode % save_freq == 0 and episode > 0:
                self.save_checkpoint(f"models/checkpoints/dqn_episode_{episode}.pth")
    
    def _action_to_idx(self, action: Action) -> int:
        """Convert Action to index for neural network"""
        shot_types = ["f", "b", "s", "v"]
        directions = [1, 2, 3]
        
        shot_type_idx = shot_types.index(action.shot_type)
        direction_idx = directions.index(action.shot_direction)
        
        return shot_type_idx * len(directions) + direction_idx
    
    def evaluate(self, episodes: int = 100) -> Dict:
        """Evaluate trained agent"""
        original_epsilon = self.agent.epsilon
        self.agent.epsilon = 0  # No exploration during evaluation
        
        rewards = []
        win_count = 0
        
        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0
            
            while True:
                action = self.agent.act(state)
                state, reward, done, info = self.env.step(action)
                total_reward += reward
                
                if done:
                    break
            
            rewards.append(total_reward)
            if total_reward > 0:  # Assuming positive reward means win
                win_count += 1
        
        # Restore original epsilon
        self.agent.epsilon = original_epsilon
        
        return {
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'win_rate': win_count / episodes,
            'total_episodes': episodes
        }
    
    def plot_training_history(self, save_path: str = None):
        """Plot training metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Episode rewards
        axes[0, 0].plot(self.training_history['episode_rewards'])
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        
        # Episode lengths
        axes[0, 1].plot(self.training_history['episode_lengths'])
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        
        # Win rates
        if self.training_history['win_rates']:
            axes[1, 0].plot(self.training_history['win_rates'])
            axes[1, 0].set_title('Win Rate (Last 100 Episodes)')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Win Rate')
        
        # Epsilon values
        axes[1, 1].plot(self.training_history['epsilon_values'])
        axes[1, 1].set_title('Exploration Rate (Epsilon)')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Epsilon')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def save_checkpoint(self, filepath: str):
        """Save model and training history"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        self.agent.save(filepath)
        
        # Save training history
        history_path = filepath.replace('.pth', '_history.json')
        import json
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f)