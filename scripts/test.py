import sys
import pathlib
from pathlib import Path
import json

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
            print(f"Match Score: Player {env.state.player_game_score}-{env.state.player_set_score} | PC {env.state.pc_game_score}-{env.state.pc_set_score}")

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
        env=env
    )  # use defaults; ensure signature matches your implementation

    ckpt_path = project_root / checkpoint
    if ckpt_path.exists():
        print(f"Loading checkpoint: {ckpt_path}")
        agent.load(str(ckpt_path))
    else:
        print(
            f"Checkpoint not found at {ckpt_path}; running agent without weights (random/initialized policy)"
        )

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
