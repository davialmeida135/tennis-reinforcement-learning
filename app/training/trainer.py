import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Optional
import json
import mlflow
import mlflow.pytorch
from app.environment.tennis_env import TennisEnv, Action, Turn
from app.agents.dqn_agent import DQNAgent


class Trainer:
    def __init__(
        self, 
        env: TennisEnv, 
        agent: DQNAgent,
        mlflow_tracking_uri: str = "https://mlflow.digi.com.br",
        experiment_name: str = "tennis-rl-dqn"
    ):
        self.env = env
        self.agent = agent
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'win_rates': [],
            'epsilon_values': [],
            'loss_values': [],
            'q_values': []
        }
        
        # MLflow setup
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(experiment_name)
    
    def train(
        self, 
        episodes: int = 1000, 
        save_freq: int = 100,
        eval_freq: int = 200,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ):
        """Train the agent for specified number of episodes with MLflow logging"""
        
        with mlflow.start_run(run_name=run_name, tags=tags):
            # Log hyperparameters
            self._log_hyperparameters(episodes, save_freq, eval_freq)
            
            # Log environment info
            self._log_environment_info()
            
            best_avg_reward = float('-inf')
            
            for episode in range(episodes):
                state = self.env.reset()
                total_reward = 0
                steps = 0
                episode_q_values = []
                episode_losses = []
                
                while True:
                    # Agent chooses action
                    action = self.agent.act(state)
                    
                    # Log Q-values for analysis
                    if hasattr(self.agent, 'q_network'):
                        import torch
                        with torch.no_grad():
                            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.agent.device)
                            q_vals = self.agent.q_network(state_tensor)
                            max_q_value = torch.max(q_vals).item()
                            episode_q_values.append(max_q_value)
                    
                    # Environment processes action
                    next_state, reward, done, info = self.env.step(action)
                    
                    # Store experience
                    action_idx = self._action_to_idx(action)
                    self.agent.remember(state, action_idx, reward, next_state, done)
                    
                    # Train the agent and capture loss
                    if len(self.agent.memory) > self.agent.batch_size:
                        loss = self.agent.replay()  # Modify replay() to return loss
                        if loss is not None:
                            episode_losses.append(loss)
                    
                    state = next_state
                    total_reward += reward
                    steps += 1
                    
                    if done:
                        break
                
                # Record training metrics
                self.training_history['episode_rewards'].append(total_reward)
                self.training_history['episode_lengths'].append(steps)
                self.training_history['epsilon_values'].append(self.agent.epsilon)
                
                # Record Q-values and losses
                if episode_q_values:
                    avg_q_value = np.mean(episode_q_values)
                    self.training_history['q_values'].append(avg_q_value)
                    mlflow.log_metric("avg_q_value", avg_q_value, step=episode)
                
                if episode_losses:
                    avg_loss = np.mean(episode_losses)
                    self.training_history['loss_values'].append(avg_loss)
                    mlflow.log_metric("avg_loss", avg_loss, step=episode)
                
                # Log episode metrics
                mlflow.log_metric("episode_reward", total_reward, step=episode)
                mlflow.log_metric("episode_length", steps, step=episode)
                mlflow.log_metric("epsilon", self.agent.epsilon, step=episode)
                
                # Calculate and log win rate (last 100 episodes)
                if episode >= 99:
                    recent_rewards = self.training_history['episode_rewards'][-100:]
                    win_rate = sum(1 for r in recent_rewards if r > 0) / len(recent_rewards)
                    avg_reward = np.mean(recent_rewards)
                    
                    self.training_history['win_rates'].append(win_rate)
                    mlflow.log_metric("win_rate_100", win_rate, step=episode)
                    mlflow.log_metric("avg_reward_100", avg_reward, step=episode)
                    
                    # Track best model
                    if avg_reward > best_avg_reward:
                        best_avg_reward = avg_reward
                        mlflow.log_metric("best_avg_reward", best_avg_reward, step=episode)
                        
                        # Save best model
                        best_model_path = f"models/best_model_episode_{episode}.pth"
                        self.save_checkpoint(best_model_path)
                        mlflow.log_artifact(best_model_path, "models")
                
                # Print progress
                if episode % 100 == 0:
                    avg_reward = np.mean(self.training_history['episode_rewards'][-100:])
                    print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Epsilon: {self.agent.epsilon:.3f}")
                
                # Periodic evaluation
                if episode % eval_freq == 0 and episode > 0:
                    eval_results = self.evaluate(episodes=50)
                    self._log_evaluation_results(eval_results, episode)
                
                # Save model checkpoint
                if episode % save_freq == 0 and episode > 0:
                    checkpoint_path = f"models/checkpoints/dqn_episode_{episode}.pth"
                    self.save_checkpoint(checkpoint_path)
                    mlflow.log_artifact(checkpoint_path, "checkpoints")
            
            # Final evaluation
            final_eval_results = self.evaluate(episodes=100)
            self._log_evaluation_results(final_eval_results, episodes, prefix="final_")
            
            # Save final training plots
            plot_path = "training_plots.png"
            self.plot_training_history(save_path=plot_path)
            mlflow.log_artifact(plot_path, "plots")
            
            # Save training history
            history_path = "training_history.json"
            with open(history_path, 'w') as f:
                json.dump(self.training_history, f)
            mlflow.log_artifact(history_path, "data")
            
            # Log final model
            final_model_path = "models/final_model.pth"
            self.save_checkpoint(final_model_path)
            mlflow.pytorch.log_model(
                pytorch_model=self.agent.q_network,
                artifact_path="final_model",
                registered_model_name="tennis-dqn-model"
            )
    
    def _log_hyperparameters(self, episodes: int, save_freq: int, eval_freq: int):
        """Log all hyperparameters to MLflow"""
        # Agent hyperparameters
        mlflow.log_param("lr", self.agent.lr)
        mlflow.log_param("gamma", self.agent.gamma)
        mlflow.log_param("epsilon_initial", 1.0)  # Assuming initial epsilon
        mlflow.log_param("epsilon_min", self.agent.epsilon_min)
        mlflow.log_param("epsilon_decay", self.agent.epsilon_decay)
        mlflow.log_param("memory_size", self.agent.memory.maxlen)
        mlflow.log_param("batch_size", self.agent.batch_size)
        mlflow.log_param("target_update_freq", self.agent.target_update_freq)
        
        # Network architecture
        mlflow.log_param("state_size", self.agent.state_size)
        mlflow.log_param("action_size", self.agent.action_size)
        mlflow.log_param("hidden_size", 128)  # Assuming default
        
        # Training parameters
        mlflow.log_param("total_episodes", episodes)
        mlflow.log_param("save_frequency", save_freq)
        mlflow.log_param("eval_frequency", eval_freq)
        
        # Environment parameters
        mlflow.log_param("point_win_reward", self.env.POINT_WIN_REWARD)
        mlflow.log_param("point_loss_penalty", self.env.POINT_LOSS_PENALTY)
        mlflow.log_param("game_win_reward", self.env.GAME_WIN_REWARD)
        mlflow.log_param("game_loss_penalty", self.env.GAME_LOSS_PENALTY)
        mlflow.log_param("set_win_reward", self.env.SET_WIN_REWARD)
        mlflow.log_param("set_loss_penalty", self.env.SET_LOSS_PENALTY)
        mlflow.log_param("illegal_action_penalty", self.env.ILLEGAL_ACTION_PENALTY)
    
    def _log_environment_info(self):
        """Log environment-specific information"""
        mlflow.log_param("action_space_size", len(self.env.action_space))
        mlflow.log_param("stroke_types", list(self.env.stroke_space.values())[:10])  # Log first 10
        mlflow.log_param("directions", self.env.direction_space)
    
    def _log_evaluation_results(self, eval_results: Dict, episode: int, prefix: str = "eval_"):
        """Log evaluation results to MLflow"""
        for metric, value in eval_results.items():
            mlflow.log_metric(f"{prefix}{metric}", value, step=episode)
    
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
        episode_lengths = []
        
        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0
            steps = 0
            
            while True:
                action = self.agent.act(state)
                state, reward, done, info = self.env.step(action)
                total_reward += reward
                steps += 1
                
                if done:
                    break
            
            rewards.append(total_reward)
            episode_lengths.append(steps)
            if total_reward > 0:
                win_count += 1
        
        # Restore original epsilon
        self.agent.epsilon = original_epsilon
        
        return {
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'win_rate': win_count / episodes,
            'avg_episode_length': np.mean(episode_lengths),
            'total_episodes': episodes
        }
    
    def plot_training_history(self, save_path: str = None):
        """Plot training metrics"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
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
            axes[0, 2].plot(self.training_history['win_rates'])
            axes[0, 2].set_title('Win Rate (Last 100 Episodes)')
            axes[0, 2].set_xlabel('Episode')
            axes[0, 2].set_ylabel('Win Rate')
        
        # Epsilon values
        axes[1, 0].plot(self.training_history['epsilon_values'])
        axes[1, 0].set_title('Exploration Rate (Epsilon)')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Epsilon')
        
        # Q-values
        if self.training_history['q_values']:
            axes[1, 1].plot(self.training_history['q_values'])
            axes[1, 1].set_title('Average Q-Values')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Q-Value')
        
        # Loss values
        if self.training_history['loss_values']:
            axes[1, 2].plot(self.training_history['loss_values'])
            axes[1, 2].set_title('Training Loss')
            axes[1, 2].set_xlabel('Episode')
            axes[1, 2].set_ylabel('Loss')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def save_checkpoint(self, filepath: str):
        """Save model and training history"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        self.agent.save(filepath)
        
        # Save training history
        history_path = filepath.replace('.pth', '_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f)