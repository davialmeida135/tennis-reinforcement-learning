from collections import defaultdict
import polars as pl
from app.models.shot import Shot

class TransitionBuilder:
    def __init__(self):
        self.unwanted_characters = {"unknown", "q", "0"}
        self.errors_and_winners = {"@", "#", "winner"}


    def build(self, df: pl.DataFrame) -> pl.DataFrame:
        self.transition_counts : defaultdict[tuple, int] = defaultdict(int)

        for shot_row in df.iter_rows(named=True):
            shot_type = shot_row["shot_type"]
            shot_direction = shot_row["shot_direction"]
            last_shot_type = shot_row["last_shot_type"]
            last_shot_direction = shot_row["last_shot_direction"]

            # Ignorar transições com valores desconhecidos ou inválidos
            if (shot_type in self.unwanted_characters or
                shot_direction in self.unwanted_characters or
                last_shot_type in self.unwanted_characters or
                last_shot_direction in self.unwanted_characters):
                continue

            # Transições entre erros e winners não são consideradas
            if (shot_type in self.errors_and_winners and
                last_shot_type in self.errors_and_winners):
                continue

            # Antes de um saque só pode vir erro ou winner
            if (last_shot_type not in self.errors_and_winners and
                shot_type == "serve"):
                continue

            # Depois de um erro ou winner só pode vir saque
            if (last_shot_type in self.errors_and_winners and
                shot_type != "serve"):
                continue

            state = (last_shot_type, last_shot_direction)
            action = (shot_type, shot_direction)
            key = (state, action)
            self.transition_counts[key] += 1

        # Save transition counts to a DataFrame
        records = []
        for (state, action), count in self.transition_counts.items():
            last_shot_type, last_shot_direction = state
            shot_type, shot_direction = action
            records.append({
                "last_shot_type": last_shot_type,
                "last_shot_direction": last_shot_direction,
                "shot_type": shot_type,
                "shot_direction": shot_direction,
                "count": count
            })

        return pl.DataFrame(records)
    

if __name__ == "__main__":
    import pathlib
    project_root = pathlib.Path(__file__).parent.parent.parent
    data_path = project_root / "data" / "processed" / "parsed_charting-m-points-2020s.csv"
    dtypes = {
        "p1_score": pl.Utf8,  # Set p1_score as string
        "p2_score": pl.Utf8,  # Set p2_score as string
        # Add other columns here if needed
    }
    df = pl.read_csv(data_path, dtypes=dtypes)
    builder = TransitionBuilder()
    transition_df = builder.build(df)
    target_path = project_root / "data" / "processed" / "shot_transitions.csv"
    transition_df.to_pandas().to_csv(target_path, index=False)