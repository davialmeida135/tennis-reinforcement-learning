import sys
import pathlib
from pathlib import Path
import json
import time

import mlflow

from app.agents.dqn_agent import DQNAgent

# add project root to path (same pattern as scripts/train.py)
project_root = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.agents.base_agent import BaseAgent
from app.data.transition_graph import TransitionBuilder
from app.environment.tennis_env import TennisEnv, Action


def load_transition_graph(path: Path):
    builder = TransitionBuilder(transitions_path=str(path), temperature=1.0)
    return builder.build()


def play_once(env: TennisEnv, agent: BaseAgent, render: bool = False):
    strokes = []
    total_reward = 0.0
    state = env.state

    while True:
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)

        # record basic info about the stroke
        stroke_record = {
            "shot_type": getattr(action, "shot_type", str(action)),
            "shot_direction": getattr(action, "shot_direction", None),
            "reward": float(reward),
            "info": info or {},
        }
        strokes.append(stroke_record)

        total_reward += reward
        state = next_state

        if render:
            print(f"Shot {len(strokes)}: {stroke_record}")
            print(f"Match Score: Player {env.state.player_set_score}-{env.state.player_game_score} |  {env.state.pc_game_score}-{env.state.pc_set_score} PC")

        time.sleep(1.5)
        if done:
            break

    result = {
        "total_reward": float(total_reward),
        "n_shots": len(strokes),
        "strokes": strokes,
        "final_info": info or {},
    }
    return result


def main(
    checkpoint: str = "models/final_model.pth",
    transitions_csv: str = "data/processed/shot_transitions_combined.csv",
    matches: int = 1,
    render: bool = True,
):
    project_root = Path(__file__).parent.parent

    transitions_path = project_root / transitions_csv
    if not transitions_path.exists():
        raise FileNotFoundError(f"Transitions file not found: {transitions_path}")

    print("Loading transition graph...")
    transition_graph = load_transition_graph(transitions_path)
    print("Creating environment...")
    env = TennisEnv(transition_graph=transition_graph, serve_first=False)

    print("Initializing agent...")
    agent = DQNAgent(
        env=env,
        epsilon=0.5
    )  # use defaults; ensure signature matches your implementation

    mlflow.set_tracking_uri("https://mlflow.digi.com.br")  # your self‑hosted URL
    model_uri = "models:/tennis-reinforce-model/None"      # name/stage in registry

    # build env first
    transition_graph = load_transition_graph(Path("data/processed/shot_transitions_combined.csv"))
    env = TennisEnv(transition_graph=transition_graph, serve_first=False)

    agent = DQNAgent(env=env, epsilon=0.0)  # epsilon=0 for inference
    agent.q_network = mlflow.pytorch.load_model(model_uri)
    agent.q_network.to(agent.device)

    results = []
    for i in range(matches):
        print(f"\n=== Playing match {i+1} ===")
        res = play_once(env, agent, render=render)
        results.append(res)
        print(
            f"Match {i+1} result: total_reward={res['total_reward']:.2f}, shots={res['n_shots']}"
        )
        # print final info if available (scores, winner flag, etc.)
        if res["final_info"]:
            print("Final info:", json.dumps(res["final_info"], indent=2))

    # optionally write a summary file
    out_path = project_root / "play_matches_results.json"
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSaved play results to {out_path}")


if __name__ == "__main__":
    main()
