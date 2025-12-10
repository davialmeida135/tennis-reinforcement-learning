import sys
import pathlib

project_root = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.environment.tennis_env import TennisEnv
from app.agents.dqn_agent import DQNAgent
from app.training.dqn_trainer import DQNTrainer
from app.data.transition_graph import TransitionBuilder


def main():
    data_path = project_root / "data" / "processed" / "shot_transitions_combined.csv"
    
    print("Building transition graph...")
    graph_builder = TransitionBuilder(transitions_path=str(data_path), temperature=1.0)
    transition_graph = graph_builder.build()
    
    print("Creating tennis environment...")
    env = TennisEnv(transition_graph=transition_graph, serve_first=True, illegal_action_penalty=-0.5)
    
    print("Initializing DQN agent...")
    agent = DQNAgent(env=env, lr=0.001, gamma=0.95)
    
    print("Initializing DQN trainer...")
    trainer = DQNTrainer(env=env, agent=agent)
    
    print("Starting training...")
    trainer.train(episodes=2000, save_freq=200, eval_freq=200, run_name="dqn_tennis_v1")


if __name__ == "__main__":
    main()