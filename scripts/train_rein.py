import sys
import pathlib

project_root = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.environment.tennis_env import TennisEnv
from app.agents.reinforce_agent import ReinforceAgent
from app.training.reinforce_trainer import ReinforceTrainer
from app.data.transition_graph import TransitionBuilder


def main():
    data_path = project_root / "data" / "processed" / "shot_transitions_combined.csv"
    
    print("Building transition graph...")
    graph_builder = TransitionBuilder(transitions_path=str(data_path), temperature=1.0)
    transition_graph = graph_builder.build()
    
    print("Creating tennis environment...")
    env = TennisEnv(transition_graph=transition_graph, serve_first=True, illegal_action_penalty=-0.5)
    
    print("Initializing REINFORCE agent...")
    agent = ReinforceAgent(env=env, lr=1e-2, gamma=0.99)
    
    print("Initializing REINFORCE trainer...")
    trainer = ReinforceTrainer(env=env, agent=agent)
    
    print("Starting training...")
    trainer.train(episodes=2000, save_freq=200, eval_freq=200, run_name="reinforce_tennis_v1")


if __name__ == "__main__":
    main()