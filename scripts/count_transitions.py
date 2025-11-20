from app.data.transition_counter import TransitionBuilder
import polars as pl
import pathlib
if __name__ == "__main__":
    files = [
        "parsed_charting-m-points-to-2009.csv",
        "parsed_charting-m-points-2010s.csv",
        "parsed_charting-m-points-2020s.csv"
    ]
    for file_name in files:
        project_root = pathlib.Path(__file__).parent.parent
        data_path = project_root / "data" / "processed" / file_name
        target_path = project_root / "data" / "processed" / f"shot_transitions_{file_name}"
        dtypes = {
            "p1_score": pl.Utf8,  # Set p1_score as string
            "p2_score": pl.Utf8,  # Set p2_score as string
            # Add other columns here if needed
        }
        df = pl.read_csv(data_path, dtypes=dtypes)
        builder = TransitionBuilder()
        transition_df = builder.build(df)
        transition_df.to_pandas().to_csv(target_path, index=False)