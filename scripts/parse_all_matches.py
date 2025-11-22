from app.data.match_parser import MatchParser
import polars as pl
import pathlib

if __name__ == "__main__":
    files = [
        "charting-m-points-to-2009.csv",
        "charting-m-points-2010s.csv",
        "charting-m-points-2020s.csv"
    ]
    for file_name in files:
        project_root = pathlib.Path(__file__).parent.parent
        data_path = project_root / "data" / "raw" / file_name
        data_path.parent.mkdir(parents=True, exist_ok=True)
        target_path = project_root / "data" / "processed" / f"parsed_{file_name}"
        target_path.parent.mkdir(parents=True, exist_ok=True)
        df = pl.read_csv(data_path)
        print(f"Parsing file: {file_name}")
        parser = MatchParser()
        parsed_all = parser.parse_all_points(df)
        parsed_all.to_pandas().to_csv(target_path, index=False)
