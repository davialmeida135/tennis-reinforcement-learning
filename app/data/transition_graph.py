import random
import numpy as np
from scipy.special import softmax
import pandas as pd
from typing import Dict, Any


def build_transition_graph():
    df = pd.read_csv(
        "data/processed/shot_transitions_parsed_charting-m-points-2010s.csv"
    )

    # Regras para remover transições ilegais
    # 1. Um "serve" não pode ser seguido de outro serve
    # 2. Um "forehand" ou "backhand" não pode ser seguido de um "serve"
    # 3. As direções devem estar entre 1 e 3
    # 4. Erros não podem ser seguidos por erros
    # 5. Um "winner" não pode ser seguido por outro "winner"
    # 6. Erros e winners não tem direção definida (usar 'unknown')

    possible_strokes = [
        "b",
        "f",
        "r",
        "i",
        "m",
        "o",
        "s",
        "v",
        "z",
        "u",
        "h",
        "l",
        "j",
        "y",
        "t",
        "k",
        "p",
    ]

    possible_types = [
        "serve",
        "@",
        "#",
        "winner",
        "b",
        "f",
        "r",
        "i",
        "m",
        "o",
        "s",
        "v",
        "z",
        "u",
        "h",
        "l",
        "j",
        "y",
        "t",
        "k",
        "p",
    ]

    errors_and_winner = ["#", "@", "winner"]

    possible_directions = [1, 2, 3]

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

                    # Transições entre erros e winners não são consideradas
                    if (s_type in errors_and_winner and
                        last_s_type in errors_and_winner):
                        continue

                    # Antes de um saque só pode vir erro ou winner
                    if (last_s_type not in errors_and_winner and
                        s_type == "serve"):
                        continue

                    # Depois de um erro ou winner só pode vir saque
                    if (last_s_type in errors_and_winner and
                        s_type != "serve"):
                        continue

                    key = (last_s_type, last_s_dir, s_type, s_dir)
                    if key not in counts_lookup:
                        counts_lookup[key] = 0

    # Build quick lookup of counts: map (last_type,last_dir,shot_type,shot_dir) -> count
    for _, row in df.iterrows():
        key = (row["last_shot_type"], row["last_shot_direction"], row["shot_type"], row["shot_direction"])
        counts_lookup[key] = int(row["count"])

    print(counts_lookup[("serve",1,"@",1)])

    graph: Dict[str, Dict[str, Dict[tuple, float]]] = {}
    records: list[dict[str, Any]] = []
    temperature = 2
    for src_type in possible_types:
        graph[src_type] = {}
        dest_types = []
        counts = []
        for src_dir in possible_directions:
            print("Looking at source:", src_type, src_dir)
            # Pegar todos os valores de contagem para transições válidas a partir de (src_type, src_dir) em counts_lookup
            for count_tuple in counts_lookup.keys():
                if str(count_tuple[0]) == str(src_type) and int(count_tuple[1]) == int(src_dir):
                    #print("  Checking count tuple:", count_tuple, counts_lookup[count_tuple])
                    dest_types.append((str(count_tuple[2]), int(count_tuple[3])))
                    counts.append(counts_lookup[count_tuple])

            logits = np.array(counts, dtype=float)

            adjusted = logits ** (1 / temperature)  # aplica temperatura
            
            probs = adjusted/adjusted.sum()

            # Construir o grafo de transição
            graph[src_type][str(src_dir)] = {}
            for i, dest in enumerate(dest_types):
                dest_type, dest_dir = dest
                prob = float(probs[i]) if i < len(probs) else 0.0
                graph[src_type][str(src_dir)][(dest_type, dest_dir)] = prob

                records.append({
                    "last_shot_type": src_type,
                    "last_shot_direction": src_dir,
                    "shot_type": dest_type,
                    "shot_direction": dest_dir,
                    "probability": prob
                })


    df_out = pd.DataFrame.from_records(records)
    df_out.to_csv(
        "data/processed/shot_transition_graph_charting-m-points.csv", index=False
    )
    # print("Source:", src_type, src_dir)
    # print(graph['f']['1'])
    return graph


print("Building transition graph...")
print(build_transition_graph())
print("Done.")