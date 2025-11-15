import polars as pl
from app.models.shot import Shot

serve_errors = ["n", "w", "d", "x", "g", "e", "V", "!", "c"]
serve_directions = ["4", "5", "6", "0"]

stroke_errors = ["n", "w", "d", "x", "!", "e", "@", "#"]
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

remover = ["7", "8", "9", "+", "=", "-", ";", "^"]

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
        shot_player=shot.shot_player,
        shot_number=shot_number,
        shot_type="",
    )

def parse_text(point: str, match_id: str, serve_player: str) -> pl.DataFrame:

    filtered_point = filter_special_characters(point)
    shots: list[Shot] = []
    shot: Shot = Shot(
        match_id=match_id,
        rally_number=1,
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

        if char in stroke_types:
            shot.shot_type = char
            continue

        if char in stroke_errors:
            shot.last_shot_type = last_shot.shot_type
            shot.last_shot_direction = last_shot.shot_direction
            shot.shot_type = f"error_{char}"
            shot.is_error = True
            last_shot = shot
            shot = append_shot(shots, last_shot, shot_number)
        elif char in winners:
            shot.last_shot_type = last_shot.shot_type
            shot.last_shot_direction = last_shot.shot_direction
            shot.shot_type = "winner"
            shot.is_winner = True
            last_shot = shot
            shot = append_shot(shots, last_shot, shot_number)
        elif char in stroke_directions:
            shot.last_shot_type = last_shot.shot_type
            shot.last_shot_direction = last_shot.shot_direction
            shot.shot_direction = char
            last_shot = shot
            shot = append_shot(shots, last_shot, shot_number)
        else:
            shot.last_shot_type = last_shot.shot_type
            shot.last_shot_direction = last_shot.shot_direction
            shot.shot_type = "unknown"
            shot.shot_direction = "unknown"
            last_shot = shot
            shot = append_shot(shots, last_shot, shot_number)

        shot_number += 1

    return pl.DataFrame(shots)

def parse_point(point: pl.DataFrame):
    point_text = point["1st"][0]
    match_id = point["match_id"][0]
    svr = point["Svr"][0]
    return parse_text(point_text, match_id, svr)


if __name__ == "__main__":
    import pathlib
    project_root = pathlib.Path(__file__).parent.parent.parent
    data_path = project_root / "data" / "raw" / "charting-m-points-2020s.csv"
    df = pl.read_csv(data_path)
    df = df[4:10]
    print(df.shape)
    print(type(df[0]))
    print(df[0])
    print(type(df[0][0]))
    print(df[0][0])
    parsed = parse_point(df)

    # parsed = parse_point(point, "match_001", "1")
    parsed.to_pandas().to_csv("parsed_point.csv", index=False)
    # print(parsed)
