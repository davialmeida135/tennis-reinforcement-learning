import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
import mlflow

from app.training.base_trainer import BaseTrainer
from app.agents.reinforce_agent import ReinforceAgent
from app.environment.tennis_env import TennisEnv


class ReinforceTrainer(BaseTrainer):
    """Trainer specifically for REINFORCE agents"""

    def __init__(
        self,
        env: TennisEnv,
        agent: ReinforceAgent,
        mlflow_tracking_uri: str = "https://mlflow.digi.com.br",
        experiment_name: str = "tennis-rl-reinforce",
    ):
        super().__init__(env, agent, mlflow_tracking_uri, experiment_name)
        self.agent: ReinforceAgent = agent

    @property
    def agent_name(self) -> str:
        return "reinforce"

    def _run_episode(self) -> Dict:
        """Run a single REINFORCE episode"""
        state = self.env.reset()
        total_reward = 0
        steps = 0
        illegal_actions = 0

        while True:
            action = self.agent.act(state)

            next_state, reward, done, info = self.env.step(action)

            # Track illegal actions
            if reward == self.env.ILLEGAL_ACTION_PENALTY:
                illegal_actions += 1

            self.agent.remember(reward)

            state = next_state
            total_reward += reward
            steps += 1

            if done:
                break

        # Update policy at the end of the episode
        loss = self.agent.update()

        return {
            "total_reward": total_reward,
            "steps": steps,
            "losses": [loss] if loss is not None else [],
            "illegal_actions": illegal_actions,
        }

    def _log_hyperparameters(self, episodes: int, save_freq: int, eval_freq: int):
        """Log REINFORCE-specific hyperparameters"""
        mlflow.log_param("agent_type", "REINFORCE")
        mlflow.log_param("lr", self.agent.lr)
        mlflow.log_param("gamma", self.agent.gamma)
        mlflow.log_param("state_size", self.agent.state_size)
        mlflow.log_param("action_size", self.agent.action_size)
        mlflow.log_param("hidden_size", self.agent.policy_network.fc1.out_features)
        mlflow.log_param("total_episodes", episodes)
        mlflow.log_param("save_frequency", save_freq)
        mlflow.log_param("eval_frequency", eval_freq)

    def _finalize_training(self, episodes: int):
        """Finalize REINFORCE training with model logging"""
        super()._finalize_training(episodes)
        
        # Log PyTorch model
        mlflow.pytorch.log_model(
            pytorch_model=self.agent.policy_network,
            artifact_path="final_model",
            registered_model_name="tennis-reinforce-model",
        )

    def plot_training_history(self, save_path: str = None):
        """Plot REINFORCE-specific training metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

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
            axes[1, 0].plot(self.training_history["win_rates"])
            axes[1, 0].set_title("Win Rate (Last 100 Episodes)")
            axes[1, 0].set_xlabel("Episode")
            axes[1, 0].set_ylabel("Win Rate")

        # Loss values
        if self.training_history["loss_values"]:
            axes[1, 1].plot(self.training_history["loss_values"])
            axes[1, 1].set_title("Policy Loss")
            axes[1, 1].set_xlabel("Episode")
            axes[1, 1].set_ylabel("Loss")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()