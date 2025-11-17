from collections import defaultdict
import polars as pl
from app.models.shot import Shot

class TransitionBuilder:
    def __init__(self):
        self.transition_counts : defaultdict[tuple, int] = defaultdict(int)


    def build(self, df: pl.DataFrame) -> pl.DataFrame:
        self.transition_counts : defaultdict[tuple, int] = defaultdict(int)

        for shot_row in df.iter_rows(named=True):
            shot_type = shot_row["shot_type"]
            shot_direction = shot_row["shot_direction"]
            last_shot_type = shot_row["last_shot_type"]
            last_shot_direction = shot_row["last_shot_direction"]
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