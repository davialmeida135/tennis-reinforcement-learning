# tennis-reinforcement-learning

Train reinforcement learning agents to play tennis rallies using a stochastic transition graph built from the Match Charting Project dataset. The environment simulates points, games, and sets, and agents learn to choose shots (type, direction) to maximize rewards.

## Installation

```sh
# Python 3.12 recommended
python -m venv .venv
. .venv/Scripts/activate   # Windows
# or: source .venv/bin/activate

pip install -r requirements.txt
```

ou

```bash
uv sync
```

## Data pipeline

1. Parse raw Match Charting datasets to shot-level CSVs:
```sh
python -m scripts.parse_all_matches
```

2. Count transitions per parsed file (enforces tennis rules, removes illegal/unknowns):
```sh
python -m scripts.count_transitions
```

3. Merge counts across files:
```sh
python -m scripts.sum_transition_counts
# outputs:
# data/processed/shot_transitions_combined.csv
# data/processed/shot_transitions_combined.parquet
```

## Quickstart (training)

```python
# minimal example
from app.data.transition_graph import TransitionBuilder
from app.environment.tennis_env import TennisEnv
from app.agents.dqn_agent import DQNAgent
from app.training.dqn_trainer import DQNTrainer

# build graph from combined counts
graph = TransitionBuilder(
    transitions_path="data/processed/shot_transitions_combined.csv",
    temperature=1.0
).build()

# create environment
env = TennisEnv(transition_graph=graph, serve_first=True, illegal_action_penalty=-0.5)

# agent & trainer
agent = DQNAgent(env=env, lr=0.001, gamma=0.95)
trainer = DQNTrainer(env=env, agent=agent)

# train with MLflow tracking
trainer.train(episodes=2000, save_freq=200, eval_freq=200, run_name="dqn_tennis_v1")
```

Or use the scripts:
```sh
# DQN
python -m scripts.train_dqn

# REINFORCE
python -m scripts.train_rein
```

## Playing matches (inference)

```sh
python -m scripts.test
```

This loads the transition graph via [`scripts.test.load_transition_graph`](scripts/test.py), initializes [`app.environment.tennis_env.TennisEnv`](app/environment/tennis_env.py), and runs matches using a loaded agent (MLflow model or local checkpoint).

## Environment overview

- State encoding: see [`app.models.env.State.encode`](app/models/env.py)
- Action space: pairs of (shot_type, direction) from `stroke_space` and `direction_space` in [`app.environment.tennis_env.TennisEnv`](app/environment/tennis_env.py)
- Illegal actions: filtered in [`app.environment.tennis_env.TennisEnv._filter_illegal_action`](app/environment/tennis_env.py), penalized by `ILLEGAL_ACTION_PENALTY`
- Match engine (scores, sets, tiebreak): [`app.environment.tennis_engine.TennisMatch`](app/environment/tennis_engine.py)

## MLflow

Training uses MLflow for metrics, parameters, and artifacts:
- Logged by [`app.training.base_trainer.BaseTrainer`](app/training/base_trainer.py)
- Models saved to [models/](models/), with JSON histories (e.g., best checkpoints like best_model_episode_1003.pth)

Configure tracking URI and experiment in trainers or scripts.

## Development

- Run unit-like experiments in notebooks:
  - [notebooks/test_methods.ipynb](notebooks/test_methods.ipynb)
  - [notebooks/transitions.ipynb](notebooks/transitions.ipynb)
- Parse and EDA:
  - [notebooks/shots.ipynb](notebooks/shots.ipynb)

## License

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br /><span xmlns:dct="http://purl.org/dc/terms/" href="http://purl.org/dc/dcmitype/Dataset" property="dct:title" rel="dct:type">Crowdsourced shot-by-shot professional tennis data</span> by <a xmlns:cc="http://creativecommons.org/ns#" href="http://www.tennisabstract.com/charting/meta.html" property="cc:attributionName" rel="cc:attributionURL">The Tennis Abstract Match Charting Project</a> is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.<br />Based on a work at <a xmlns:dct="http://purl.org/dc/terms/" href="https://github.com/JeffSackmann/tennis_MatchChartingProject" rel="dct:source">https://github.com/JeffSackmann/tennis_MatchChartingProject</a>.

Attribution is required. Non-commercial use only.
