from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional
import json
import mlflow
import mlflow.pytorch

from app.agents.base_agent import BaseAgent
from app.environment.tennis_env import TennisEnv, Action


class BaseTrainer(ABC):
    """Base class for all trainers"""

    def __init__(
        self,
        env: TennisEnv,
        agent: BaseAgent,
        mlflow_tracking_uri: str = "https://mlflow.digi.com.br",
        experiment_name: str = "tennis-rl",
    ):
        self.env = env
        self.agent = agent
        self.training_history = {
            "episode_rewards": [],
            "episode_lengths": [],
            "win_rates": [],
            "loss_values": [],
            "illegal_actions": [],
        }

        self._action_to_idx_map = {
            (shot_type, direction): idx
            for idx, (shot_type, direction) in enumerate(self.env.action_space)
        }

        # MLflow setup
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(experiment_name)

    def train(
        self,
        episodes: int = 100,
        save_freq: int = 100,
        eval_freq: int = 200,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ):
        """Main training loop"""
        with mlflow.start_run(run_name=run_name, tags=tags):
            self._log_hyperparameters(episodes, save_freq, eval_freq)
            self._log_environment_info()

            best_avg_reward = float("-inf")

            for episode in range(episodes):
                episode_data = self._run_episode()
                
                # Record metrics
                self._record_episode_metrics(episode_data, episode)
                
                # Log to MLflow
                self._log_episode_to_mlflow(episode_data, episode)

                # Calculate and log rolling metrics
                if episode >= 99:
                    best_avg_reward = self._log_rolling_metrics(episode, best_avg_reward)

                # Print progress
                if episode % 100 == 0:
                    self._print_progress(episode)

                # Periodic evaluation
                if episode % eval_freq == 0 and episode > 0:
                    eval_results = self.evaluate(episodes=50)
                    self._log_evaluation_results(eval_results, episode)

                # Save checkpoints
                if episode % save_freq == 0 and episode > 0:
                    checkpoint_path = f"models/checkpoints/{self.agent_name}_episode_{episode}.pth"
                    self.save_checkpoint(checkpoint_path)
                    mlflow.log_artifact(checkpoint_path, "checkpoints")

            # Final steps
            self._finalize_training(episodes)

    @abstractmethod
    def _run_episode(self) -> Dict:
        """Run a single episode and return metrics"""
        pass

    @abstractmethod
    def _log_hyperparameters(self, episodes: int, save_freq: int, eval_freq: int):
        """Log agent-specific hyperparameters"""
        pass

    @property
    @abstractmethod
    def agent_name(self) -> str:
        """Return the agent name for file naming"""
        pass

    def _record_episode_metrics(self, episode_data: Dict, episode: int):
        """Record episode metrics to training history"""
        self.training_history["episode_rewards"].append(episode_data["total_reward"])
        self.training_history["episode_lengths"].append(episode_data["steps"])
        self.training_history["illegal_actions"].append(episode_data["illegal_actions"])
        
        if episode_data.get("losses"):
            avg_loss = np.mean(episode_data["losses"])
            self.training_history["loss_values"].append(avg_loss)

    def _log_episode_to_mlflow(self, episode_data: Dict, episode: int):
        """Log episode metrics to MLflow"""
        mlflow.log_metric("episode_reward", episode_data["total_reward"], step=episode)
        mlflow.log_metric("episode_length", episode_data["steps"], step=episode)
        mlflow.log_metric("illegal_actions", episode_data["illegal_actions"], step=episode)
        
        if episode_data.get("losses"):
            mlflow.log_metric("avg_loss", np.mean(episode_data["losses"]), step=episode)

    def _log_rolling_metrics(self, episode: int, best_avg_reward: float) -> float:
        """Calculate and log rolling window metrics"""
        recent_rewards = self.training_history["episode_rewards"][-100:]
        win_rate = sum(1 for r in recent_rewards if r > 0) / len(recent_rewards)
        avg_reward = np.mean(recent_rewards)

        self.training_history["win_rates"].append(win_rate)
        mlflow.log_metric("win_rate_100", win_rate, step=episode)
        mlflow.log_metric("avg_reward_100", avg_reward, step=episode)

        # Track best model
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            mlflow.log_metric("best_avg_reward", best_avg_reward, step=episode)
            
            best_model_path = f"models/best_model_episode_{episode}.pth"
            self.save_checkpoint(best_model_path)
            mlflow.log_artifact(best_model_path, "models")
        
        return best_avg_reward

    def _print_progress(self, episode: int):
        """Print training progress"""
        avg_reward = np.mean(self.training_history["episode_rewards"][-100:])
        print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}")

    def _log_environment_info(self):
        """Log environment-specific information"""
        mlflow.log_param("action_space_size", len(self.env.action_space))
        mlflow.log_param("stroke_types", list(self.env.stroke_space.values())[:10])
        mlflow.log_param("directions", self.env.direction_space)
        mlflow.log_param("point_win_reward", self.env.POINT_WIN_REWARD)
        mlflow.log_param("point_loss_penalty", self.env.POINT_LOSS_PENALTY)
        mlflow.log_param("game_win_reward", self.env.GAME_WIN_REWARD)
        mlflow.log_param("game_loss_penalty", self.env.GAME_LOSS_PENALTY)
        mlflow.log_param("set_win_reward", self.env.SET_WIN_REWARD)
        mlflow.log_param("set_loss_penalty", self.env.SET_LOSS_PENALTY)
        mlflow.log_param("illegal_action_penalty", self.env.ILLEGAL_ACTION_PENALTY)

    def _log_evaluation_results(self, eval_results: Dict, episode: int, prefix: str = "eval_"):
        """Log evaluation results to MLflow"""
        for metric, value in eval_results.items():
            mlflow.log_metric(f"{prefix}{metric}", value, step=episode)

    def _action_to_idx(self, action: Action) -> int:
        """Convert Action to index for neural network"""
        key = (action.shot_type, action.shot_direction)
        try:
            return self._action_to_idx_map[key]
        except KeyError:
            raise ValueError(
                f"Invalid action: {action}. "
                f"Shot type '{action.shot_type}' with direction '{action.shot_direction}' "
                f"is not in the environment's action space."
            )

    def evaluate(self, episodes: int = 100) -> Dict:
        """Evaluate trained agent"""
        rewards = []
        win_count = 0
        episode_lengths = []

        for _ in range(episodes):
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

        return {
            "avg_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "win_rate": win_count / episodes,
            "avg_episode_length": np.mean(episode_lengths),
            "total_episodes": episodes,
        }

    def _finalize_training(self, episodes: int):
        """Finalize training: evaluation, plots, and model saving"""
        # Final evaluation
        final_eval_results = self.evaluate(episodes=100)
        self._log_evaluation_results(final_eval_results, episodes, prefix="final_")

        # Save final training plots
        plot_path = "training_plots.png"
        self.plot_training_history(save_path=plot_path)
        mlflow.log_artifact(plot_path, "plots")

        # Save training history
        history_path = "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.training_history, f)
        mlflow.log_artifact(history_path, "data")

        # Save final model
        final_model_path = "models/final_model.pth"
        self.save_checkpoint(final_model_path)

    @abstractmethod
    def plot_training_history(self, save_path: str = None):
        """Plot training metrics"""
        pass

    def save_checkpoint(self, filepath: str):
        """Save model and training history"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        self.agent.save(filepath)

        history_path = filepath.replace(".pth", "_history.json")
        with open(history_path, "w") as f:
            json.dump(self.training_history, f)