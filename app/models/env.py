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
        return 7
    
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
        """Convert state tuple to numerical encoding"""
        # state_tuple = (last_shot_type, last_shot_direction, player_game_score, 
        #                player_set_score, pc_game_score, pc_set_score, player_serves)
                
        # Encode shot type (map to integer)
        shot_type_encoding = env.reverse_stroke_space.get(self.last_shot_type, 0)
        
        # Encode shot direction
        shot_dir_encoding = self.last_shot_direction if isinstance(self.last_shot_direction, int) else int(self.last_shot_direction)
        
        # Encode game scores (convert tennis scores to numbers)
        game_score_map = {"0": 0, "15": 1, "30": 2, "40": 3, "AD": 4}
        p_game_encoding = game_score_map.get(self.player_game_score, 0)
        pc_game_encoding = game_score_map.get(self.pc_game_score, 0)
        
        # Serves is already boolean (convert to 0/1)
        serves_encoding = 1 if self.player_serves else 0
        
        return [
            shot_type_encoding,
            shot_dir_encoding,
            p_game_encoding,
            self.player_set_score,
            pc_game_encoding,
            self.pc_set_score,
            serves_encoding
        ]