import numpy as np
from scipy.special import softmax
import pandas as pd
from typing import Dict, Any


def build_transition_graph():
    df = pd.read_csv("data/processed/shot_transitions_parsed_charting-m-points-2010s.csv")

    # Regras para remover transições ilegais
    # 1. Um "serve" não pode ser seguido de outro serve
    # 2. Um "forehand" ou "backhand" não pode ser seguido de um "serve"
    # 3. As direções devem estar entre 1 e 3
    # 4. Erros não podem ser seguidos por erros
    # 5. Um "winner" não pode ser seguido por outro "winner"
    # 6. Erros e winners não tem direção definida (usar 'unknown')

    possible_stroke_results = ['b', '@', 'f', 'r', '#', 'i', 'm', 'o',
        'winner', 's', 'v', 'z', 'u', 'h', 'l', 'j', 'y', 't', 'k', 'p',
        'q','unknown']

    possible_types = ['serve', 'b', '@', 'f', 'r', '#', 'i', 'm', 'o',
        'winner', 's', 'v', 'z', 'u', 'h', 'l', 'j', 'y', 't', 'k', 'p',
        'q','unknown']

    erros = ["#", "@"]

    possible_directions = ['unknown','1','2','3','0']

    from scipy.special import softmax
    counts_lookup = {}

    # Para cada tipo/direção possível, garantir que exista uma entrada no dicionário
    for s_type in possible_types:
        for s_dir in possible_directions:
            # Garantir que todas as combinações de last_s_type e last_s_dir existam
            for last_s_type in possible_types:
                for last_s_dir in possible_directions:
                    if last_s_type == "serve" and s_type == "serve":
                        continue  # Regra 1
                    if last_s_type not in erros and s_type == "serve":
                        continue  # Regra 2
                    if last_s_type in erros + ["winner"]:
                        last_s_dir = 'unknown'  # Corrigir direção ilegal
                    if s_type in erros + ["winner"]:
                        s_dir = 'unknown'  # Corrigir direção ilegal

                    key = (last_s_type, last_s_dir, s_type, s_dir)
                    if key not in counts_lookup:
                        counts_lookup[key] = 0


    # Build quick lookup of counts: map (last_type,last_dir,shot_type,shot_dir) -> count
    # Replace direction_4,5,6 with 1,2,3 respectively for serves
    for _, row in df.iterrows():
        if row["last_shot_type"] == "serve" and row["last_shot_direction"] in ["direction_4", "direction_5", "direction_6"]:
            last_dir = str(int(row["last_shot_direction"].split("_")[1]) - 3)
        else:
            last_dir = row["last_shot_direction"]
        if row["shot_type"] == "serve" and row["shot_direction"] in ["direction_4", "direction_5", "direction_6"]:
            direction = str(int(row["shot_direction"].split("_")[1]) - 3)
        else:
            direction = row["shot_direction"]

        if direction not in ["1", "2", "3", "unknown"] or last_dir not in ["1", "2", "3", "unknown"]:
            continue
        key = (row["last_shot_type"], last_dir, row["shot_type"], direction)
        counts_lookup[key] = int(row["count"])

    normal_target_pairs = [(t, d) for t in possible_stroke_results for d in possible_directions]

    error_target_pairs = [(t, d) for t in ["serve"] for d in possible_directions]

    for idx, pair in enumerate(normal_target_pairs):
        if pair[0] in erros + ["winner"] and pair[1] != 'unknown':
            normal_target_pairs[idx] = (pair[0], 'unknown')


    graph: Dict[str, Dict[str, Dict[tuple, float]]] = {}
    records: list[dict[str, Any]] = []
    temperature = 1.0
    for src_type in possible_types:
        graph[src_type] = {}
        for src_dir in possible_directions:

            counts = [counts_lookup.get((src_type, src_dir, t, d), 0) for (t, d) in normal_target_pairs]
            if src_type in erros + ["winner"]:
                src_dir = 'unknown'  # Corrigir direção ilegal
                counts = [counts_lookup.get((src_type, src_dir, t, d), 0) for (t, d) in error_target_pairs]

            # gather counts in consistent order
            # apply softmax (with temperature)
            if temperature != 1.0:
                logits = np.array(counts) / float(temperature)
            else:
                logits = np.array(counts, dtype=float)
            probs = softmax(logits) if logits.size > 0 else np.array([])

            graph[src_type][src_dir] = {}
            for (t, d), c, p in zip(normal_target_pairs if src_type not in erros + ["winner"] else error_target_pairs, counts, probs):
                graph[src_type][src_dir][(t,d)] = float(p)

                records.append({
                    "last_shot_type": src_type,
                    "last_shot_direction": src_dir,
                    "shot_type": t,
                    "shot_direction": d,
                    "count": int(c),
                    "probability": float(p)
                })


    df_out = pd.DataFrame.from_records(records)
    df_out.to_csv("data/processed/shot_transition_graph_charting-m-points.csv", index=False)
    #print("Source:", src_type, src_dir)
    #print(graph['f']['1'])
    return graph
