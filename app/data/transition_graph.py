# Import necessary libraries
import pandas as pd
from typing import Dict, Tuple, Any, Optional
from pathlib import Path
import numpy as np

# Import softmax function
from scipy.special import softmax

possible_types = ['serve', 'b', '@', 'f', 'unknown', 'r', '#', 'i', 'm', 'o',
       'winner', 's', 'v', 'z', 'u', 'h', 'l', 'j', 'y', 't', 'k', 'p',
       'q']

possible_directions = ['direction_4', '1', '3', 'unknown', 'direction_5', '2',
       'direction_6', 'direction_0']


def remove_illegal_transitions(
    counts_lookup: Dict[tuple, int]
) -> Dict[tuple, int]:
    """
    Remove illegal transitions from the transition graph by setting their probabilities to zero.

    """
    error_and_winner_types = {"@", "#", "winner"}
    normalized_counts: dict[tuple, int] = {}
    for (lt, ld, t, d), c in counts_lookup.items():
        if t in error_and_winner_types and d != "unknown":
            # aggregate into (t, 'unknown') bucket
            nk = (lt, ld, t, "unknown")
            normalized_counts[nk] = normalized_counts.get(nk, 0) + c
        else:
            nk = (lt, ld, t, d)
            normalized_counts[nk] = normalized_counts.get(nk, 0) + c
    return normalized_counts

def build_transition_graph_from_counts(
    counts_df: Optional[pd.DataFrame] = None,
    counts_csv: Optional[str] = None,
    temperature: float = 1.0
) -> Tuple[Dict[str, Dict[str, Dict[str, Dict[str, float]]]], pd.DataFrame]:
    """
    Build transition probability graph from a counts DataFrame or CSV.

    - counts_df: DataFrame with columns
        ['last_shot_type','last_shot_direction','shot_type','shot_direction','count']
    - counts_csv: path to CSV in [data/processed](data/processed) (used if counts_df is None).
    - temperature: softmax temperature (higher -> more uniform).

    Returns:
      - nested dict: graph[last_type][last_dir][shot_type][shot_dir] = probability
      - flat DataFrame with columns:
        ['last_shot_type','last_shot_direction','shot_type','shot_direction','count','probability']
    """
    if counts_df is None:
        if counts_csv is None:
            project_root = Path(__file__).parent.parent.parent
            counts_csv = project_root / "data" / "processed" / "shot_transitions.csv"
        counts_df = pd.read_csv(counts_csv)

    # ensure expected columns
    expected = {"last_shot_type","last_shot_direction","shot_type","shot_direction","count"}
    if not expected.issubset(set(counts_df.columns)):
        raise ValueError(f"counts_df must contain columns: {expected}")

    # Build quick lookup of counts: map (last_type,last_dir,shot_type,shot_dir) -> count
    counts_lookup = {}
    for _, r in counts_df.iterrows():
        key = (r["last_shot_type"], r["last_shot_direction"], r["shot_type"], r["shot_direction"])
        counts_lookup[key] = int(r["count"])

    counts_lookup = remove_illegal_transitions(counts_lookup)

    graph: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    records: list[dict[str, Any]] = []

    # For each possible source (last_type, last_dir) create action vector across all targets
    target_pairs = [(t, d) for t in possible_types for d in possible_directions]
    for src_type in possible_types:
        graph[src_type] = {}
        for src_dir in possible_directions:
            # gather counts in consistent order
            counts = [counts_lookup.get((src_type, src_dir, t, d), 0) for (t, d) in target_pairs]

            # apply softmax (with temperature)
            if temperature != 1.0:
                logits = np.array(counts) / float(temperature)
            else:
                logits = np.array(counts, dtype=float)
            probs = softmax(logits) if logits.size > 0 else np.array([])

            graph[src_type][src_dir] = {}
            for (t, d), c, p in zip(target_pairs, counts, probs):
                if t not in graph[src_type][src_dir]:
                    graph[src_type][src_dir][t] = {}
                graph[src_type][src_dir][t][d] = float(p)

                records.append({
                    "last_shot_type": src_type,
                    "last_shot_direction": src_dir,
                    "shot_type": t,
                    "shot_direction": d,
                    "count": int(c),
                    "probability": float(p)
                })

    df_out = pd.DataFrame.from_records(records)
    return graph, df_out


if __name__ == "__main__":
    # Example CLI: writes probabilities to data/processed/shot_transition_probs.csv
    project_root = Path(__file__).parent.parent.parent
    counts_path = project_root / "data" / "processed" / "shot_transitions_parsed_charting-m-points-2010s.csv"
    out_path = project_root / "data" / "processed" / "shot_transition_probs.csv"

    graph, probs_df = build_transition_graph_from_counts(counts_csv=counts_path)
    probs_df.to_csv(out_path, index=False)
    print(f"Wrote transition probabilities to {out_path}")
