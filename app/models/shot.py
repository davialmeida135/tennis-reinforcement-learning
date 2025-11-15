from pydantic import BaseModel


class Shot(BaseModel):
    match_id: str
    rally_number: int
    serve_player: int
    shot_player: int
    shot_number: int
    last_shot_type: str = "#"
    last_shot_direction: str = "unknown"
    shot_type: str
    shot_direction: str = "unknown"
    p1_score: str = "0"
    p2_score: str = "0"
    p1_set: int = 0
    p2_set: int = 0
    is_serve: bool = False
    is_winner: bool = False
    is_error: bool = False