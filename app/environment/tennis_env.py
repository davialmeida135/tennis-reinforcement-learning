import time
from typing import Dict, Tuple, Optional
from app.environment.tennis_engine import TennisMatch
from app.models.env import Action, State, Turn
import random


class TennisEnv:
    def __init__(
        self,
        transition_graph: Dict[str, Dict[str, Dict[tuple, float]]],
        serve_first: bool = True,
        point_win_reward: int = 1,
        point_loss_penalty: int = -1,
        game_win_reward: int = 10,
        game_loss_penalty: int = -10,
        set_win_reward: int = 50,
        set_loss_penalty: int = -50,
        base_penalty: float = -0.0,
        illegal_action_penalty: int = -20,
    ):
        self.POINT_WIN_REWARD = point_win_reward
        self.POINT_LOSS_PENALTY = point_loss_penalty
        self.GAME_WIN_REWARD = game_win_reward
        self.GAME_LOSS_PENALTY = game_loss_penalty
        self.SET_WIN_REWARD = set_win_reward
        self.SET_LOSS_PENALTY = set_loss_penalty
        self.BASE_PENALTY = base_penalty
        self.ILLEGAL_ACTION_PENALTY = illegal_action_penalty
        
        self.stroke_space = {
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
        self.reverse_stroke_space = (
            {v: k for k, v in self.stroke_space.items()} if self.stroke_space else None
        )
        self.direction_space = [1, 2, 3]

        self.action_space = [
            (shot_type, shot_direction) for shot_type in self.stroke_space.values() for shot_direction in self.direction_space
        ]

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
        self.match = TennisMatch()
        self.match.start_match()

    def reset(self):
        # TODO
        self.state: State = None
        self.match = TennisMatch()

    def step(self, action:tuple[str, int]) -> Tuple[State, int, bool, dict]:

        reward = 0
        done = False

        action = Action(shot_type=action[0], shot_direction=action[1])
        is_illegal = self._filter_illegal_action(action)
        if is_illegal:
            print("Ação ilegal detectada:", action)
            return self.state, self.ILLEGAL_ACTION_PENALTY, False, {}
        
        # Aplica ação do jogador
        self._update_state(action)
        # Simula até acabar a rodada do PC
        print(f"Vez do {self.turn}")
        # Sample next state
        action = Action(
            shot_type=self.state.last_shot_type,
            shot_direction=self.state.last_shot_direction,
        )
        next_actions = self._choose_next_2_actions(action)

        new_state, new_reward = self._compute_actions(next_actions)

        reward += new_reward

        if new_state is not None:
            self.state.player_game_score = new_state[0]
            self.state.pc_game_score = new_state[1]
            self.state.player_set_score = new_state[2]
            self.state.pc_set_score = new_state[3]

        while self.turn == Turn.PC:
            action = Action(
                shot_type=self.state.last_shot_type,
                shot_direction=self.state.last_shot_direction,
            )
            next_actions = self._choose_next_2_actions(action)
            self._update_state(next_actions[0])
            new_state, new_reward = self._compute_actions(next_actions)
            reward += new_reward

            if new_state is not None:
                self.state.player_game_score = new_state[0]
                self.state.pc_game_score = new_state[1]
                self.state.player_set_score = new_state[2]
                self.state.pc_set_score = new_state[3]

        # Apply action to the environment and update state
        info = {}
        self.turn = Turn.PLAYER
        return self.state, reward, done, info

    def _filter_illegal_action(self, action: Action) -> bool:
        # Verifica se a ação é legal no estado atual
        if self.state.last_shot_type in self.errors or self.state.last_shot_type in self.winners:
            if action.shot_type != "serve":
                return True  # Apenas saque é permitido após erro ou winner
            
        if self.state.last_shot_type in self.stroke_space.values():
            if action.shot_type == "serve":
                return True  # Saque não é permitido após saque ou golpe

        return False

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
            return None

        if player_scored:
            print("Ponto para o PLAYER")
            (
                current_game_p1,
                current_game_p2,
                current_set_p1,
                current_set_p2,
                game_winner,
                set_winner,
            ) = self.match.point(player=Turn.PLAYER)
        else:
            print("Ponto para o PC")
            (
                current_game_p1,
                current_game_p2,
                current_set_p1,
                current_set_p2,
                game_winner,
                set_winner,
            ) = self.match.point(player=Turn.PC)

        if game_winner is not None:
            # Switch server for new game
            self.server = Turn.PLAYER if self.server == Turn.PC else Turn.PC
            self.state.player_serves = self.server == Turn.PLAYER
            print(f"Game ended! Server switched to: {self.server}")

        # Passa a vez para o sacador
        self.first_serve = True
        self.turn = Turn.PLAYER if self.server == Turn.PLAYER else Turn.PC
        print(f"Turno para: {self.turn}")

        # Atualiza o placar do estado
        self.state.player_game_score = current_game_p1
        self.state.pc_game_score = current_game_p2
        self.state.player_set_score = current_set_p1
        self.state.pc_set_score = current_set_p2

        return (
            current_game_p1,
            current_game_p2,
            current_set_p1,
            current_set_p2,
            game_winner,
            set_winner,
        )

    def _get_reward(self, state, player_scored: bool) -> int:
        reward = self.BASE_PENALTY

        if not state:
            return reward

        game_winner = state[4]
        set_winner = state[5]

        if player_scored:
            reward += self.POINT_WIN_REWARD
            if game_winner == Turn.PLAYER:
                reward += self.GAME_WIN_REWARD
            if set_winner == Turn.PLAYER:
                reward += self.SET_WIN_REWARD
        else:
            reward += self.POINT_LOSS_PENALTY
            if game_winner == Turn.PC:
                reward += self.GAME_LOSS_PENALTY
            if set_winner == Turn.PC:
                reward += self.SET_LOSS_PENALTY

        return reward

    def _compute_actions(self, next_actions: list[Action]) -> State:
        # Primeira ação é ou um erro/winner do jogador ou um lance normal do pc
        is_serve = self.state.last_shot_type == "serve"
        if next_actions[0].shot_type in self.errors:
            if self.turn == Turn.PLAYER:
                # Erro do player, PC scores
                print("Player errou")
                new_state = self._update_score(player_scored=False, is_serve=is_serve)
                new_reward = self._get_reward(new_state, player_scored=False)
                self._update_state(next_actions[0])
                return new_state, new_reward
            elif self.turn == Turn.PC:
                # Erro do PC, Player scores
                print("PC errou")
                new_state = self._update_score(player_scored=True, is_serve=is_serve)
                new_reward = self._get_reward(new_state, player_scored=True)
                self._update_state(next_actions[0])
                return new_state, new_reward

        elif next_actions[0].shot_type in self.winners:
            if self.turn == Turn.PLAYER:
                # Player made a winner, Player scores
                print("Player fez um winner")
                new_state = self._update_score(player_scored=True)
                new_reward = self._get_reward(new_state, player_scored=True)
                self._update_state(next_actions[0])
                return new_state, new_reward
            elif self.turn == Turn.PC:
                # PC made a winner, PC scores
                print("PC fez um winner")
                new_state = self._update_score(player_scored=False)
                new_reward = self._get_reward(new_state, player_scored=False)
                self._update_state(next_actions[0])
                return new_state, new_reward

        self._update_state(next_actions[0])
        is_serve = self.state.last_shot_type == "serve"

        # Se chegou aqui, é um lance normal do PC, verificar se o PC errou no segundo lance
        self.turn = Turn.PC

        if next_actions[1].shot_type in self.errors:
            # Erro do Player, PC scores
            print("PC errou seu lance no segundo lance")
            new_state = self._update_score(player_scored=True, is_serve=is_serve)
            new_reward = self._get_reward(new_state, player_scored=True)
            self._update_state(next_actions[1])
            return new_state, new_reward
        # Agora ver se o PC fez um winner no segundo lance
        elif next_actions[1].shot_type in self.winners:
            # Player made a winner, Player scores
            print("PC fez um winner no segundo lance")
            new_state = self._update_score(player_scored=False)
            new_reward = self._get_reward(new_state, player_scored=False)
            self._update_state(next_actions[1])
            return new_state, new_reward

        # Se chegou aqui, o ponto continua
        self._update_state(next_actions[0])
        self.turn = Turn.PLAYER
        print("Ponto continua, turno de:", self.turn)
        return None, self.BASE_PENALTY

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
    print(env.action_space)

    start = time.time()
    next_state, reward, done, info = env.step(("serve", 1))
    end = time.time()
    print(f"Step took {end - start} seconds")
    # env.step(("f", "2"))
    # env.step(("b", "1"))
    # env.step(("@", "unknown"))
    print(f"State: {next_state},\nReward: {reward},\nDone: {done},\nInfo: {info}")
