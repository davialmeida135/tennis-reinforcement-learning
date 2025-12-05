import polars as pl
from pathlib import Path

def main():
    project_root = Path(__file__).parent.parent
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # match files produced by scripts/count_transitions.py
    files = [p for p in processed_dir.glob("shot_transitions_parsed*") if p.is_file()]

    if not files:
        print("No shot_transitions_parsed files found in", processed_dir)
        return

    dfs = []
    schema_overrides = {
        "last_shot_direction": pl.Utf8,
        "shot_direction": pl.Utf8,
    }

    for f in files:
        try:
            df = pl.read_csv(f, schema_overrides=schema_overrides)
        except Exception as e:
            # fallback: try reading with pandas then convert
            import pandas as pd
            print(f"polars failed to read {f}, trying pandas ({e})")
            df = pl.from_pandas(pd.read_csv(f, dtype=str))
        # normalize column names
        df = df.rename(
            {
                c: c.strip()
                for c in df.columns
            }
        )
        # ensure required cols exist
        required = ["last_shot_type", "last_shot_direction", "shot_type", "shot_direction", "count"]
        if not all(col in df.columns for col in required):
            print(f"Skipping {f} - missing required columns. Found: {df.columns}")
            continue

        # coerce count to integer
        df = df.with_columns([
            pl.col("count").cast(pl.Int64).fill_null(0)
        ])
        dfs.append(df.select(required))

    if not dfs:
        print("No valid transition files to aggregate.")
        return

    combined: pl.DataFrame = pl.concat(dfs, how="vertical")

    # group and sum counts
    agg = (
        combined.group_by(
            ["last_shot_type", "last_shot_direction", "shot_type", "shot_direction"]
        )
        .agg(pl.col("count").sum().alias("count"))
        .sort("count", descending=True)
    )

    # output paths
    out_csv = processed_dir / "shot_transitions_combined.csv"
    out_parquet = processed_dir / "shot_transitions_combined.parquet"

    agg.to_pandas().to_csv(out_csv, index=False)
    agg.write_parquet(out_parquet)

    print(f"Processed {len(files)} files.")
    print(f"Unique transitions: {agg.height}")
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_parquet}")

if __name__ == "__main__":
    main()