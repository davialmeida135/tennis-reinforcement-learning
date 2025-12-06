from enum import Enum
from pydantic import BaseModel
from typing import List, Optional

class Turn(Enum):
    PLAYER = 1
    PC = 0


class Action(BaseModel):
    shot_type: str
    shot_direction: int

    def to_tuple(self, types_dict, directions_dict) -> tuple:
        return (types_dict[self.shot_type], directions_dict[self.shot_direction])


class State(BaseModel):
    last_shot_type: str
    last_shot_direction: int
    player_game_score: str
    player_set_score: int
    pc_game_score: str
    pc_set_score: int
    player_serves: bool

    def __len__(self):
        return 21 + 3 + 1 + 1 + 3  # Total length of encoded state
    
    def to_tuple(self) -> tuple:
        return (
            self.last_shot_type,
            self.last_shot_direction,
            self.player_game_score,
            self.player_set_score,
            self.pc_game_score,
            self.pc_set_score,
            self.player_serves
        )
    
    def encode(self, env) -> List[int]:
        """Convert state to numerical encoding with one-hot encoding for shots and directions"""
        
        # One-hot encode shot type
        num_shot_types = len(env.last_shot_space)
        shot_type_onehot = [0] * num_shot_types
        shot_type_idx = env.reverse_last_shot_space.get(self.last_shot_type, 0)
        shot_type_onehot[shot_type_idx] = 1
        
        # One-hot encode shot direction (directions are 1, 2, 3)
        num_directions = len(env.direction_space)
        direction_onehot = [0] * num_directions
        # Convert direction to 0-indexed
        direction_idx = env.direction_space.index(self.last_shot_direction) if self.last_shot_direction in env.direction_space else 0
        direction_onehot[direction_idx] = 1
        
        # One-hot encode player game score
        game_score_map = {"0": 0, "15": 1, "30": 2, "40": 3, "AD": 4}
        p_game_encoding = game_score_map.get(self.player_game_score, 0)
        player_game = [p_game_encoding]
        
        # One-hot encode PC game score
        pc_game_encoding = game_score_map.get(self.pc_game_score, 0)
        pc_game = [pc_game_encoding]
        
        # Set scores (raw values, sets can go from 0-7 theoretically)
        player_set_encoding = self.player_set_score
        pc_set_encoding = self.pc_set_score
        
        # Serves encoding (binary)
        serves_encoding = 1 if self.player_serves else 0
        
        # Concatenate all features
        encoded_state = (
            shot_type_onehot +      # Length: num_shot_types (21 for serve + all strokes)
            direction_onehot +       # Length: 3
            player_game +     # Length: 1
            pc_game +         # Length: 1
            [player_set_encoding, pc_set_encoding, serves_encoding]  # Length: 3
        )
        
        return encoded_state