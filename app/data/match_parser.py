import polars as pl
from app.models.shot import Shot


class MatchParser:
    def __init__(self):
        self.serve_errors = ["n", "w", "d", "x", "g", "e", "V", "!", "c"]
        self.serve_directions = {"4": "1", "5": "2", "6": "3", "0": "0"}
        self.stroke_errors = ["@", "#"]
        self.stroke_directions = ["1", "2", "3", "0"]
        self.stroke_types = [
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
        self.winners = ["*"]
        self.remover = ["7", "8", "9", "+", "=", "-", ";", "^"] + self.serve_errors

        self.goal_columns = [
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

        self.shot: Shot = Shot(
            match_id="0000",
            rally_number=1,
            serve_player=1,
            shot_player=1,
            shot_number=1,
            shot_type="",
        )

    def filter_special_characters(self, point: str) -> str:
        return point.translate({ord(c): None for c in self.remover})

    def append_shot(self, shots: list[Shot], shot: Shot, shot_number: int) -> Shot:
        shots.append(shot.model_dump())
        shot.shot_player = 3 - shot.shot_player  # Alterna entre jogador 1 e 2
        shot.shot_number = shot_number
        shot.last_shot_type = shot.shot_type
        shot.last_shot_direction = shot.shot_direction
        shot.shot_type = "unknown"
        shot.shot_direction = "unknown"
        shot.is_serve = False
        shot.is_winner = False
        shot.is_error = False

        return shot

    def parse_point(self, point: pl.DataFrame) -> pl.DataFrame:
        if not point["2nd"][0]:
            point_text = point["1st"][0]
            second_serve = False
        else:
            point_text = point["2nd"][0]
            second_serve = True

        filtered_point = self.filter_special_characters(point_text)
        shots: list[Shot] = []

        # Ao iniciar o ponto, configurar os atributos básicos do golpe
        # Mantém valores do golpe anterior para last_shot_type e last_shot_direction
        self.shot.match_id = point["match_id"][0]
        self.shot.rally_number = point["Pt"][0]
        self.shot.p1_score = point["Pts"][0].split("-")[0]
        self.shot.p2_score = point["Pts"][0].split("-")[1]
        self.shot.p1_games = int(point["Gm1"][0]) if point["Gm1"][0] is not None else 0
        self.shot.p2_games = int(point["Gm2"][0]) if point["Gm2"][0] is not None else 0
        self.shot.p1_sets = point["Set1"][0]
        self.shot.p2_sets = point["Set2"][0]
        self.shot.serve_player = point["Svr"][0]
        self.shot.shot_player = point["Svr"][0]
        self.shot.shot_number = 1
        self.shot.shot_type = ""
        self.shot.second_serve = second_serve
        self.shot.full_text = point_text

        shot_number = 1
        for i, char in enumerate(filtered_point):

            # Tratamento especial para o primeiro golpe (saque)
            if shot_number == 1:
                self.shot.is_serve = True
                self.shot.shot_type = "serve"
                self.shot.shot_direction = (
                    char
                    if char in self.serve_errors
                    else self.serve_directions.get(char, "unknown")
                )
                self.shot = self.append_shot(shots, self.shot, shot_number)
                shot_number += 1

                continue

            # Capturar o tipo do golpe, mas não avançar o número do golpe ainda
            if char in self.stroke_types:
                self.shot.shot_type = char
                continue
            elif char in self.stroke_errors:

                self.shot.shot_type = f"{char}"
                self.shot.is_error = True

                if not self.shot.last_shot_type == "serve":
                    self.shot.shot_player = 3 - self.shot.shot_player
                self.shot.shot_direction = self.shot.last_shot_direction

                self.shot = self.append_shot(shots, self.shot, shot_number)
            elif char in self.winners:
                self.shot.shot_type = "winner"
                self.shot.shot_direction = self.shot.last_shot_direction
                self.shot.is_winner = True
                self.shot.shot_player = 3 - self.shot.shot_player

                self.shot = self.append_shot(shots, self.shot, shot_number)
            elif char in self.stroke_directions:
                self.shot.shot_direction = char

                self.shot = self.append_shot(shots, self.shot, shot_number)
            else:
                self.shot.shot_type = "unknown"
                self.shot.shot_direction = "unknown"

                self.shot = self.append_shot(shots, self.shot, shot_number)

            shot_number += 1

        return pl.DataFrame(shots)

    def parse_all_points(self, df: pl.DataFrame) -> pl.DataFrame:
        """Parse all points in the DataFrame and return a single concatenated DataFrame."""
        all_shots = []

        for i in range(len(df)):
            point_row = df[i]
            print(f"Parsing point {i+1}/{len(df)}")
            parsed_point = self.parse_point(point_row)
            if parsed_point.is_empty():
                continue
            all_shots.append(parsed_point)

        # Concatenate all parsed points into a single DataFrame
        return pl.concat(all_shots)


if __name__ == "__main__":
    import pathlib

    project_root = pathlib.Path(__file__).parent.parent.parent
    data_path = project_root / "data" / "raw" / "points-reduced.csv"
    target_path = project_root / "data" / "processed" / "parsed_points.csv"
    df = pl.read_csv(data_path)

    parser = MatchParser()
    parsed_all = parser.parse_all_points(df)
    parsed_all.to_pandas().to_csv(target_path, index=False)
