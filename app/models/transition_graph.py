from pydantic import BaseModel
from typing import Optional

class Transition(BaseModel):
    from_shot:str
    from_direction:str
    to_shot:str
    to_direction:str
    probability:float