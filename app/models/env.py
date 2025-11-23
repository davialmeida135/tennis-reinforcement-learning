from pydantic import BaseModel
from typing import Optional

class Action(BaseModel):
    shot_type: str
    shot_direction: str

    def to_tuple(self, types_dict, directions_dict) -> tuple:
        return (types_dict[self.shot_type], directions_dict[self.shot_direction])


class State(BaseModel):
    last_shot_type: str
    last_shot_direction: str
    player_game_score: str
    player_set_score: int
    pc_game_score: str
    pc_set_score: int
    player_serves: bool
    
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