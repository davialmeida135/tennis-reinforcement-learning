from typing import Dict, Tuple, Optional
from app.models.env import Action, State
import random


class TennisEnv:
    def __init__(
        self,
        transition_graph: Dict[str, Dict[str, Dict[tuple, float]]],
        serve_first: bool = True,
    ):
        self.action_space = {
            0: "serve",
            1: "b",
            2: "f",
            3: "r",
            4: "i",
            5: "m",
            6: "o",
            7: "s",
            8: "v",
            9: "p",
            10: "z",
            11: "u",
            12: "h",
            13: "l",
            14: "j",
            15: "y",
            16: "t",
            17: "k",
        }
        self.reverse_action_space = (
            {v: k for k, v in self.action_space.items()} if self.action_space else None
        )
        self.direction_space = {0: "1", 1: "2", 2: "3"}
        self.reverse_direction_space = (
            {v: k for k, v in self.direction_space.items()}
            if self.direction_space
            else None
        )
        self.errors = {"@", "#"}
        self.winners = {"winner"}

        if serve_first:
            initial_shot_type = "#"
            initial_shot_direction = "unknown"
            self.turn = 0  # Player's turn
        else:
            initial_shot_type = "serve"
            initial_shot_direction = random.choice(self.direction_space)
            self.turn = 1  # PC's turn

        self.state: State = State(
            last_shot_type=initial_shot_type,
            last_shot_direction=initial_shot_direction,
            player_game_score="0",
            player_set_score=0,
            pc_game_score="0",
            pc_set_score=0,
            player_serves=serve_first,
        )
        self.transition_graph = transition_graph

    def reset(self):
        # TODO
        self.state: State = None

    def step(self, action):

        reward = 0
        
        action = Action(shot_type=action[0], shot_direction=action[1])
        # Simula até acabar a rodada do PC
        while self.state.turn == 1:
            # Sample next state
            next_action = self._choose_next_action(action)

            # Update environment state
            self.state.last_shot_type = next_action.shot_type
            self.state.last_shot_direction = next_action.shot_direction

            self._compute_score()
            reward += 1

        # Apply action to the environment and update state
        reward = 0
        done = False
        info = {}
        return self.state, reward, done, info

    def _choose_next_action(self, action: Action) -> Action:
        possible_next_actions = self.transition_graph.get(action.shot_type, {}).get(
            action.shot_direction, {}
        )
        if possible_next_actions is None:
            raise ValueError("No possible transitions from current state.")

        # Sample next state based on transition probabilities
        # Build lists of candidates and their probabilities
        candidates = list(possible_next_actions.keys())
        # print(possible_next_actions)
        probs = list(possible_next_actions.values())

        if not candidates:
            raise ValueError("No transitions available from given action.")

        total = sum(probs)
        if total <= 0:
            raise ValueError("Transition probabilities sum to zero.")

        # Sample next state
        next_shot_type, next_shot_direction = random.choices(
            candidates, weights=probs, k=1
        )[0]

        print(f"Chosen next action: ({next_shot_type}, {next_shot_direction})")

        return Action(shot_type=next_shot_type, shot_direction=next_shot_direction)

    def _choose_next_2_actions(self, action: Action) -> Action:
        executed_actions = []
        for _ in range(2):
            next_action = self._choose_next_action(action)
            executed_actions.append(next_action)
            action = next_action
        return executed_actions

    def _compute_score(self) -> State:
        # Logic to compute the next state based on the current state
        # Compute game score, set score, turn, etc. based on errors, winners
        self.state.turn = 0 if self.state.turn == 1 else 1
        pass

    def sample_action(self) -> Action:
        shot_type = random.choice(list(self.transition_graph.keys()))
        shot_direction = random.choice(self.direction_space)
        return Action(shot_type=shot_type, shot_direction=shot_direction)


if __name__ == "__main__":
    # Example usage
    from app.data.transition_graph_2 import build_transition_graph
    import random

    transition_graph = build_transition_graph()
    print(transition_graph)
    env = TennisEnv(transition_graph)
    next_state, reward, done, info = env.step(("f", "unknown"))
    env.step(("f", "2"))
    env.step(("b", "1"))
    env.step(("@", "unknown"))
    print(next_state, reward, done, info)
