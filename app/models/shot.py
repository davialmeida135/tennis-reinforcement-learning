from pydantic import BaseModel
from typing import Optional


class Shot(BaseModel):
    """
    Classe que representa um golpe (shot) em uma partida de tênis.
    Attributes:
        match_id (str): ID da partida.
        rally_number (int): Número do rali dentro do ponto.
        serve_player (int): Jogador que está sacando.
        shot_player (int): Jogador que está executando o golpe.
        shot_number (int): Número do golpe dentro do rali.
        last_shot_type (str): Tipo do último golpe.
        last_shot_direction (str): Direção do último golpe.
        shot_type (str): Tipo do golpe atual.
        shot_direction (str): Direção do golpe atual.
        p1_score (str): Placar do jogador 1.
        p2_score (str): Placar do jogador 2.
        p1_set (int): Número de sets ganhos pelo jogador 1.
        p2_set (int): Número de sets ganhos pelo jogador 2.
        is_serve (bool): Indica se o golpe é um saque.
        is_winner (bool): Indica se o golpe é um winner.
        is_error (bool): Indica se o golpe é um erro não forçado.
    """
    match_id: str
    rally_number: int
    serve_player: int
    shot_player: int
    shot_number: int
    last_shot_type: str = "#"
    last_shot_direction: str = "1"
    shot_type: str
    shot_direction: str = "unknown"
    p1_score: str = "0"
    p2_score: str = "0"
    p1_games: int = 0
    p2_games: int = 0
    p1_sets: int = 0
    p2_sets: int = 0
    is_serve: bool = False
    is_winner: bool = False
    is_error: bool = False
    full_text: Optional[str] = None
    second_serve: Optional[bool] = None

class Point(BaseModel):
    match_id: str
    pt: int = 0  # Pt
    set1: int = 0  # Set1
    set2: int = 0  # Set2
    gm1: int = 0  # Gm1
    gm2: int = 0  # Gm2
    pts: str = "0"  # Pts (e.g. "15-30")
    gm_number: int = 0  # Gm#
    tb_set: bool = False  # TbSet (tiebreak set indicator)
    server: Optional[int] = None  # Svr (server player id)
    first_serve: Optional[float] = None  # 1st (first-serve stat, % or speed)
    second_serve: Optional[float] = None  # 2nd (second-serve stat)
    notes: str = ""  # Notes
    PtWinner: Optional[int] = None  # PtWinner (player id)

    class Config:
        schema_extra = {
            "description": "Point model. Field comments show original CSV headers: "
                           "Pt, Set1, Set2, Gm1, Gm2, Pts, Gm#, TbSet, Svr, 1st, 2nd, Notes, PtWinner"
        }
