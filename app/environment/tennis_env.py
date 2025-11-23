from enum import Enum
import time
from typing import Dict, Tuple, Optional
from app.models.env import Action, State
import random


class Turn(Enum):
    PLAYER = 1
    PC = 0


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
        self.direction_space = [1, 2, 3]

        self.errors = {"@", "#"}
        self.winners = {"winner"}

        self.first_serve = True

        if serve_first:
            initial_shot_type = "#"
            initial_shot_direction = random.choice(self.direction_space)
            self.turn = Turn.PLAYER  # Player's turn
            self.server = Turn.PLAYER
        else:
            initial_shot_type = "serve"
            initial_shot_direction = random.choice(self.direction_space)
            self.turn = Turn.PC  # PC's turn
            self.server = Turn.PC

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
        self._update_state(action)
        # Simula até acabar a rodada do PC
        print(f"Vez do {self.turn}")
        # Sample next state
        action = Action(
            shot_type=self.state.last_shot_type,
            shot_direction=self.state.last_shot_direction,
        )
        next_actions = self._choose_next_2_actions(action)

        self._compute_score(next_actions)

        while self.turn == Turn.PC:
            action = Action(
                shot_type=self.state.last_shot_type,
                shot_direction=self.state.last_shot_direction,
            )
            next_actions = self._choose_next_2_actions(action)
            self._update_state(next_actions[0])
            self._compute_score(next_actions)

        reward += 1

        # Apply action to the environment and update state
        reward = 0
        done = False
        info = {}
        self.turn = Turn.PLAYER
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

        # FIXME: DADOS MOCKADOS PARA TESTE
        # next_shot_type, next_shot_direction = random.choices(
        #     [("#", "3"), ("f","1"), ("b","2"), ("winner","1")], k=1
        # )[0]

        # if action.shot_type in self.errors or action.shot_type in self.winners:
        #     next_shot_direction = random.choice(["1", "2", "3"])
        #     next_shot_type = "serve"

        print(f"Chosen next action: ({next_shot_type}, {next_shot_direction})")

        return Action(shot_type=next_shot_type, shot_direction=next_shot_direction)

    def _choose_next_2_actions(self, action: Action) -> Action:
        executed_actions = []
        action = Action(
            shot_type=self.state.last_shot_type,
            shot_direction=self.state.last_shot_direction,
        )
        for _ in range(2):
            next_action = self._choose_next_action(action)
            executed_actions.append(next_action)
            action = next_action
        return executed_actions

    def _update_score(self, player_scored: bool, is_serve: Optional[bool] = False):

        if is_serve and self.first_serve:
            self.first_serve = False
            print("Primeiro saque perdido, segunda chance.")
            return

        game_ended = random.choices([True, False], weights=[0.5, 0.5])[
            0
        ]  # Mocked for testing

        if player_scored:
            print("Ponto para o PLAYER")
        else:
            print("Ponto para o PC")
        if game_ended:
            # Reset scores for new game
            self.state.player_game_score = "0"
            self.state.pc_game_score = "0"
            # Update set scores accordingly
            if player_scored:
                self.state.player_set_score += 1
            else:
                self.state.pc_set_score += 1

            # Switch server for new game
            self.server = Turn.PLAYER if self.server == Turn.PC else Turn.PC
            print(f"Game ended! Server switched to: {self.server}")

        # Passa a vez para o sacador
        self.first_serve = True
        self.turn = Turn.PLAYER if self.server == Turn.PLAYER else Turn.PC

        print(f"Turno para: {self.turn}")

    def _compute_score(self, next_actions: list[Action]) -> State:
        # Primeira ação é ou um erro/winner do jogador ou um lance normal do pc
        is_serve = self.state.last_shot_type == "serve"
        if next_actions[0].shot_type in self.errors:
            if self.turn == Turn.PLAYER:
                # Erro do player, PC scores
                print("Player errou")
                self._update_score(player_scored=False, is_serve=is_serve)
                self._update_state(next_actions[0])
                return
            elif self.turn == Turn.PC:
                # Erro do PC, Player scores
                print("PC errou")
                self._update_score(player_scored=True, is_serve=is_serve)
                self._update_state(next_actions[0])
                return
        elif next_actions[0].shot_type in self.winners:
            if self.turn == Turn.PLAYER:
                # Player made a winner, Player scores
                print("Player fez um winner")
                self._update_score(player_scored=True)
                self._update_state(next_actions[0])
                return
            elif self.turn == Turn.PC:
                # PC made a winner, PC scores
                print("PC fez um winner")
                self._update_score(player_scored=False)
                self._update_state(next_actions[0])
                return

        self._update_state(next_actions[0])
        # Se chegou aqui, é um lance normal do PC, verificar se o PC errou no segundo lance
        self.turn = Turn.PC

        if next_actions[1].shot_type in self.errors:
            # Erro do Player, PC scores
            print("PC errou seu lance no segundo lance")
            self._update_score(player_scored=True, is_serve=is_serve)
            self._update_state(next_actions[1])
            return
        # Agora ver se o PC fez um winner no segundo lance
        elif next_actions[1].shot_type in self.winners:
            # Player made a winner, Player scores
            print("PC fez um winner no segundo lance")
            self._update_score(player_scored=False)
            self._update_state(next_actions[1])
            return

        # Se chegou aqui, o ponto continua
        self._update_state(next_actions[0])
        self.turn = Turn.PLAYER
        print("Ponto continua, turno de:", self.turn)
        return

    def _update_state(self, action: Action):
        self.state.last_shot_type = action.shot_type
        self.state.last_shot_direction = action.shot_direction

    def sample_action(self) -> Action:
        shot_type = random.choice(list(self.transition_graph.keys()))
        shot_direction = random.choice(self.direction_space)
        return Action(shot_type=shot_type, shot_direction=shot_direction)


if __name__ == "__main__":
    # Example usage
    from app.data.transition_graph import TransitionBuilder
    import random
    import pathlib

    project_root = pathlib.Path(__file__).parent.parent.parent
    data_path = (
        project_root
        / "data"
        / "processed"
        / "shot_transitions_parsed_charting-m-points-2020s.csv"
    )

    start = time.time()
    graph_builder = TransitionBuilder(transitions_path=str(data_path))
    transition_graph = graph_builder.build()
    end = time.time()
    
    print(f"Transition graph built in {end - start} seconds")
    # print(transition_graph)
    env = TennisEnv(transition_graph, serve_first=True)

    start = time.time()
    next_state, reward, done, info = env.step(("serve", 1))
    end = time.time()
    print(f"Step took {end - start} seconds")
    # env.step(("f", "2"))
    # env.step(("b", "1"))
    # env.step(("@", "unknown"))
    print(next_state, reward, done, info)
