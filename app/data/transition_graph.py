import random
import numpy as np
from scipy.special import softmax
import pandas as pd
from typing import Dict, Any


class TransitionBuilder:
    def __init__(self, transitions_path: str = None, temperature: float = 1.0):
        self.transitions_path = transitions_path
        self.temperature = temperature
        self.df = pd.read_csv(transitions_path)
        self.transition_counts: Dict[tuple, int] = {}

        self.errors_and_winners = {"#", "@", "winner"}
        self.possible_types = [
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
        self.possible_directions = [1, 2, 3]

    def _build_transition_dict(
        self,
        ):
        counts_lookup = {}

        # Para cada tipo/direção possível, garantir que exista uma entrada no dicionário
        for s_type in self.possible_types:
            for s_dir in self.possible_directions:
                # Garantir que todas as combinações de last_s_type e last_s_dir existam
                for last_s_type in self.possible_types:
                    for last_s_dir in self.possible_directions:
                        if last_s_type == "serve" and s_type == "serve":
                            continue  # Regra 1

                        # Transições entre erros e winners não são consideradas
                        if s_type in self.errors_and_winners and last_s_type in self.errors_and_winners:
                            continue

                        # Antes de um saque só pode vir erro ou winner
                        if last_s_type not in self.errors_and_winners and s_type == "serve":
                            continue

                        # Depois de um erro ou winner só pode vir saque
                        if last_s_type in self.errors_and_winners and s_type != "serve":
                            continue

                        key = (last_s_type, last_s_dir, s_type, s_dir)
                        if key not in counts_lookup:
                            counts_lookup[key] = 0
        # Build quick lookup of counts: map (last_type,last_dir,shot_type,shot_dir) -> count
        for _, row in self.df.iterrows():
            key = (
                row["last_shot_type"],
                row["last_shot_direction"],
                row["shot_type"],
                row["shot_direction"],
            )
            counts_lookup[key] = int(row["count"])
        
        return counts_lookup
    
    def _get_transition_counts(self, counts_lookup, src_type, src_dir):
        dest_types = []
        counts = []

        for count_tuple in counts_lookup.keys():
            if str(count_tuple[0]) == str(src_type) and int(count_tuple[1]) == int(
                src_dir
            ):
                # print("  Checking count tuple:", count_tuple, counts_lookup[count_tuple])
                dest_types.append((str(count_tuple[2]), int(count_tuple[3])))
                counts.append(counts_lookup[count_tuple])
        
        return dest_types, counts

    def _calculate_probabilities(self, counts: list[int], temperature: float) -> np.ndarray:
        logits = np.array(counts, dtype=float)

        adjusted = logits ** (1 / temperature)  # aplica temperatura

        probs = adjusted / adjusted.sum()
        return probs

    def build(self) -> pd.DataFrame:
        counts_lookup = self._build_transition_dict()

        graph: Dict[str, Dict[str, Dict[tuple, float]]] = {}
        records: list[dict[str, Any]] = []
        for src_type in self.possible_types:
            graph[src_type] = {}
            dest_types = []
            counts = []
            for src_dir in self.possible_directions:
                # Pegar todos os valores de contagem para transições 
                # válidas a partir de (src_type, src_dir) em counts_lookup
                dest_types, counts = self._get_transition_counts(counts_lookup, src_type, src_dir)

                probs = self._calculate_probabilities(counts, self.temperature)

                # Construir o grafo de transição
                graph[src_type][str(src_dir)] = {}
                for i, dest in enumerate(dest_types):
                    dest_type, dest_dir = dest
                    prob = float(probs[i]) if i < len(probs) else 0.0
                    graph[src_type][str(src_dir)][(dest_type, dest_dir)] = prob

                    records.append(
                        {
                            "last_shot_type": src_type,
                            "last_shot_direction": src_dir,
                            "shot_type": dest_type,
                            "shot_direction": dest_dir,
                            "probability": prob,
                        }
                    )

        return graph

if __name__ == "__main__":
    import pathlib

    project_root = pathlib.Path(__file__).parent.parent.parent
    data_path = project_root / "data" / "processed" / "shot_transitions_parsed_charting-m-points-2020s.csv"

    def build_transition_graph():
        builder = TransitionBuilder(
            transitions_path=str(data_path),
            temperature=1.0
        )
        graph = builder.build()
        return graph