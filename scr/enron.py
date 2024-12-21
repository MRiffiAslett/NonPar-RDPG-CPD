import pandas as pd
import numpy as np
import requests
from io import StringIO

def load_enron_data():
    data_url = "https://www.cis.jhu.edu/~parky/Enron/execs.email.linesnum.topic"
    response = requests.get(data_url)
    response.raise_for_status()
    raw_data_text = response.text

    raw_data_cleaned = raw_data_text.replace('"', '').strip()
    lines = raw_data_cleaned.split("\n")
    valid_lines = [line for line in lines if len(line.split()) == 5]
    filtered_data = "\n".join(valid_lines)
    filtered_data_io = StringIO(filtered_data)

    df = pd.read_csv(filtered_data_io, sep="\s+", names=["time", "sender", "receiver", "topic"])
    df["time"] = pd.to_datetime(df["time"], unit='s')
    df["time_group"] = df["time"].dt.to_period("W")

    all_nodes = pd.unique(df[["sender", "receiver"]].values.ravel())
    node_map = {node: idx for idx, node in enumerate(all_nodes)}
    num_nodes = len(all_nodes)

    time_groups = df.groupby("time_group")
    adjacency_data = []

    for group, data in time_groups:
        adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

        for _, row in data.iterrows():
            sender_idx = node_map[row["sender"]]
            receiver_idx = node_map[row["receiver"]]
            adj_matrix[sender_idx, receiver_idx] = 1
            adj_matrix[receiver_idx, sender_idx] = 1

        adjacency_data.append({
            "time_group": group,
            "adjacency_matrix": adj_matrix
        })

    num_timepoints = len(adjacency_data)
    adjacency_matrices = np.zeros((num_nodes, num_nodes, num_timepoints), dtype=int)

    for t, entry in enumerate(adjacency_data):
        adjacency_matrices[:, :, t] = entry["adjacency_matrix"]

    return adjacency_matrices
