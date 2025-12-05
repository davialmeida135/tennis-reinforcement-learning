import sys
import pathlib
from pathlib import Path

# Add project root to Python path
project_root = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.environment.tennis_env import TennisEnv
from app.agents.dqn_agent import DQNAgent
from app.training.trainer import Trainer
from app.data.transition_graph import TransitionBuilder


def main():
    """Main training script for tennis RL agent"""
    
    # Set up paths
    data_path = project_root / "data" / "processed" / "shot_transitions_combined.csv"
    
    print("Building transition graph...")
    # Build transition graph
    graph_builder = TransitionBuilder(
        transitions_path=str(data_path),
        temperature=1.0
    )
    transition_graph = graph_builder.build()
    print("Transition graph built successfully!")
    
    # Create environment
    print("Creating tennis environment...")
    env = TennisEnv(
        transition_graph=transition_graph,
        serve_first=True,
        
    )
    
    # Create DQN agent
    print("Initializing DQN agent...")
    agent = DQNAgent(
        env=env,
        lr=0.001,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        memory_size=10000,
        batch_size=32,
        target_update_freq=100
    )
    
    # Create trainer
    trainer = Trainer(
        env=env,
        agent=agent,
        mlflow_tracking_uri="https://mlflow.digi.com.br",
        experiment_name="tennis-rl-dqn"
    )
    
    # Training configuration
    training_config = {
        "episodes": 2000,
        "save_freq": 100,
        "eval_freq": 200,
        "run_name": "dqn_tennis_v1",
        "tags": {
            "model_type": "DQN",
            "environment": "tennis",
            "data_source": "charting-m-points-2020s",
            "temperature": "1.0"
        }
    }
    
    print(f"Starting training for {training_config['episodes']} episodes...")
    
    # Start training
    trainer.train(**training_config)
    
    print("Training completed!")
    
    # Final evaluation
    print("Running final evaluation...")
    final_results = trainer.evaluate(episodes=100)
    print(f"Final evaluation results:")
    print(f"  Average reward: {final_results['avg_reward']:.2f}")
    print(f"  Win rate: {final_results['win_rate']:.2%}")
    print(f"  Average episode length: {final_results['avg_episode_length']:.1f}")
    
    # Save final plots
    trainer.plot_training_history(save_path="training_results.png")
    print("Training plots saved to training_results.png")


if __name__ == "__main__":
    main()