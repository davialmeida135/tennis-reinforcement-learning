import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
import mlflow

from app.training.base_trainer import BaseTrainer
from app.agents.dqn_agent import DQNAgent
from app.environment.tennis_env import TennisEnv


class DQNTrainer(BaseTrainer):
    """Trainer specifically for DQN agents"""

    def __init__(
        self,
        env: TennisEnv,
        agent: DQNAgent,
        mlflow_tracking_uri: str = "https://mlflow.digi.com.br",
        experiment_name: str = "tennis-rl-dqn",
    ):
        super().__init__(env, agent, mlflow_tracking_uri, experiment_name)
        self.agent: DQNAgent = agent
        
        self.training_history["epsilon_values"] = []
        self.training_history["q_values"] = []

    @property
    def agent_name(self) -> str:
        return "dqn"

    def _run_episode(self) -> Dict:
        """Run a single DQN episode"""
        state = self.env.reset()
        total_reward = 0
        steps = 0
        episode_losses = []
        episode_q_values = []
        illegal_actions = 0

        while True:
            action = self.agent.act(state)

            next_state, reward, done, info = self.env.step(action)

            # Track illegal actions
            if reward == self.env.ILLEGAL_ACTION_PENALTY:
                illegal_actions += 1

            # Store experience
            action_idx = self._action_to_idx(action)
            self.agent.remember(state, action_idx, reward, next_state, done)

            # Train the agent
            if len(self.agent.memory) > self.agent.batch_size:
                loss = self.agent.replay()
                if loss is not None:
                    episode_losses.append(loss)

            state = next_state
            total_reward += reward
            steps += 1

            if done:
                break

        return {
            "total_reward": total_reward,
            "steps": steps,
            "losses": episode_losses,
            "q_values": episode_q_values,
            "illegal_actions": illegal_actions,
            "epsilon": self.agent.epsilon,
        }

    def _record_episode_metrics(self, episode_data: Dict, episode: int):
        """Record DQN-specific metrics"""
        super()._record_episode_metrics(episode_data, episode)
        
        self.training_history["epsilon_values"].append(episode_data["epsilon"])
        
        if episode_data.get("q_values"):
            avg_q_value = np.mean(episode_data["q_values"])
            self.training_history["q_values"].append(avg_q_value)

    def _log_episode_to_mlflow(self, episode_data: Dict, episode: int):
        """Log DQN-specific metrics to MLflow"""
        super()._log_episode_to_mlflow(episode_data, episode)
        
        mlflow.log_metric("epsilon", episode_data["epsilon"], step=episode)
        
        if episode_data.get("q_values"):
            mlflow.log_metric("avg_q_value", np.mean(episode_data["q_values"]), step=episode)

    def _log_hyperparameters(self, episodes: int, save_freq: int, eval_freq: int):
        """Log DQN-specific hyperparameters"""
        mlflow.log_param("agent_type", "DQN")
        mlflow.log_param("lr", self.agent.lr)
        mlflow.log_param("gamma", self.agent.gamma)
        mlflow.log_param("epsilon_initial", 1.0)
        mlflow.log_param("epsilon_min", self.agent.epsilon_min)
        mlflow.log_param("epsilon_decay", self.agent.epsilon_decay)
        mlflow.log_param("memory_size", self.agent.memory.maxlen)
        mlflow.log_param("batch_size", self.agent.batch_size)
        mlflow.log_param("target_update_freq", self.agent.target_update_freq)
        mlflow.log_param("state_size", self.agent.state_size)
        mlflow.log_param("action_size", self.agent.action_size)
        mlflow.log_param("hidden_size", self.agent.q_network.fc1.out_features)
        mlflow.log_param("total_episodes", episodes)
        mlflow.log_param("save_frequency", save_freq)
        mlflow.log_param("eval_frequency", eval_freq)

    def _print_progress(self, episode: int):
        """Print DQN-specific progress"""
        avg_reward = np.mean(self.training_history["episode_rewards"][-100:])
        print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Epsilon: {self.agent.epsilon:.3f}")

    def _finalize_training(self, episodes: int):
        """Finalize DQN training with model logging"""
        super()._finalize_training(episodes)
        
        mlflow.pytorch.log_model(
            pytorch_model=self.agent.q_network,
            artifact_path="final_model",
            registered_model_name="tennis-dqn-model",
        )

    def plot_training_history(self, save_path: str = None):
        """Plot DQN-specific training metrics"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Episode rewards
        axes[0, 0].plot(self.training_history["episode_rewards"])
        axes[0, 0].set_title("Episode Rewards")
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Reward")

        # Episode lengths
        axes[0, 1].plot(self.training_history["episode_lengths"])
        axes[0, 1].set_title("Episode Lengths")
        axes[0, 1].set_xlabel("Episode")
        axes[0, 1].set_ylabel("Steps")

        # Win rates
        if self.training_history["win_rates"]:
            axes[0, 2].plot(self.training_history["win_rates"])
            axes[0, 2].set_title("Win Rate (Last 100 Episodes)")
            axes[0, 2].set_xlabel("Episode")
            axes[0, 2].set_ylabel("Win Rate")

        # Epsilon values
        axes[1, 0].plot(self.training_history["epsilon_values"])
        axes[1, 0].set_title("Exploration Rate (Epsilon)")
        axes[1, 0].set_xlabel("Episode")
        axes[1, 0].set_ylabel("Epsilon")

        # Q-values
        if self.training_history["q_values"]:
            axes[1, 1].plot(self.training_history["q_values"])
            axes[1, 1].set_title("Average Q-Values")
            axes[1, 1].set_xlabel("Episode")
            axes[1, 1].set_ylabel("Q-Value")

        # Loss values
        if self.training_history["loss_values"]:
            axes[1, 2].plot(self.training_history["loss_values"])
            axes[1, 2].set_title("Training Loss")
            axes[1, 2].set_xlabel("Episode")
            axes[1, 2].set_ylabel("Loss")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()