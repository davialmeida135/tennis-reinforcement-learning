import polars as pl
from app.models.shot import Shot

serve_errors = ["n", "w", "d", "x", "g", "e", "V", "!", "c"]
serve_directions = ["4", "5", "6", "0"]

stroke_errors = ["@", "#"]
stroke_directions = ["1", "2", "3", "0"]
stroke_types = [
    "f",
    "b",
    "r",
    "s",
    "v",
    "z",
    "o",
    "p",
    "u",
    "y",
    "l",
    "m",
    "h",
    "i",
    "j",
    "k",
    "t",
    "q",
]

winners = ["*"]

remover = ["7", "8", "9", "+", "=", "-", ";", "^", "n", "w", "d", "x", "!", "e", "g","e", "V", "!", "c"]

goal_columns = [
    "match_id",
    "point_id",
    "game_score",
    "set_score",
    "previous_shot_type",
    "previous_shot_direction",
    "player",
    "rally_number",
    "shot_number",
    "shot_type",
    "shot_direction",
    "is_winner",
    "is_error",
    "is_serve",
]

point = "4f18b1f1b1b3f3f1s1f+3f1z1*"


def filter_special_characters(point: str) -> str:
    return point.translate({ord(c): None for c in remover})


def append_shot(shots: list[Shot], shot: Shot, shot_number: int) -> Shot:
    shots.append(shot.model_dump())
    shot_number += 1
    return Shot(
        match_id=shot.match_id,
        rally_number=shot.rally_number,
        serve_player=shot.serve_player,
        shot_player=shot.shot_player,
        shot_number=shot_number,
        shot_type="",
    )

def _parse_point(point: str, match_id: str, serve_player: str) -> pl.DataFrame:

    filtered_point = filter_special_characters(point)
    shots: list[Shot] = []
    shot: Shot = Shot(
        match_id=match_id,
        rally_number=1,
        serve_player=serve_player,
        shot_player=serve_player,
        shot_number=1,
        shot_type="",
    )
    shot_number = 1
    for i, char in enumerate(filtered_point):

        if shot_number == 1:
            shot.is_serve = True
            shot.shot_type = "serve"
            shot.shot_direction = (
                "error_" + char
                if char in serve_errors
                else "direction_" + char if char in serve_directions else "unknown"
            )
            last_shot = shot
            shot = append_shot(shots, last_shot, shot_number)
            shot_number += 1
            continue

        shot.is_serve = False

        # Capturar o tipo do golpe, mas não avançar o número do golpe ainda
        if char in stroke_types:
            shot.shot_type = char
            continue

        # Símbolos que denotam fim do golpe
        shot.last_shot_type = last_shot.shot_type
        shot.last_shot_direction = last_shot.shot_direction

        if char in stroke_errors:
            shot.shot_type = f"error_{char}"
            shot.is_error = True
            last_shot = shot
            shot = append_shot(shots, last_shot, shot_number)
        elif char in winners:
            shot.shot_type = "winner"
            shot.is_winner = True
            last_shot = shot
            shot = append_shot(shots, last_shot, shot_number)
        elif char in stroke_directions:
            shot.shot_direction = char
            last_shot = shot
            shot = append_shot(shots, last_shot, shot_number)
        else:
            shot.shot_type = "unknown"
            shot.shot_direction = "unknown"
            last_shot = shot
            shot = append_shot(shots, last_shot, shot_number)

        shot_number += 1

    return pl.DataFrame(shots)

def parse_point(point: pl.DataFrame):
    if not point["2nd"][0]:
        point_text = point["1st"][0]
    else:
        point_text = point["2nd"][0]
    match_id = point["match_id"][0]
    svr = point["Svr"][0]
    return _parse_point(point_text, match_id, svr)

def parse_all_points(df: pl.DataFrame) -> pl.DataFrame:
    """Parse all points in the DataFrame and return a single concatenated DataFrame."""
    all_shots = []
    
    for i in range(len(df)):
        point_row = df[i]
        parsed_point = parse_point(point_row)
        if parsed_point.is_empty():
            continue
        all_shots.append(parsed_point)
    
    # Concatenate all parsed points into a single DataFrame
    return pl.concat(all_shots)

if __name__ == "__main__":
    import pathlib
    project_root = pathlib.Path(__file__).parent.parent.parent
    data_path = project_root / "data" / "raw" / "charting-m-points-2020s.csv"
    df = pl.read_csv(data_path)
    #df = df[4:10]

    parsed_all = parse_all_points(df)
    parsed_all.to_pandas().to_csv("parsed_points.csv", index=False)
    print(f"Parsed {len(df)} points into {len(parsed_all)} shots")
    "4b2f3f2f2n@"
    # parsed = parse_point(point, "match_001", "1")
    # print(parsed)
