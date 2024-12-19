import pandas as pd
import numpy as np
import requests
from io import StringIO

def load_enron_data():
    """
    Downloads and preprocesses the Enron dataset, returning a 3D numpy array
    of adjacency matrices and the total number of unique nodes.

    Returns:
        adjacency_matrices (numpy.ndarray): A 3D numpy array of shape (num_nodes, num_nodes, num_timepoints).
        num_nodes (int): Total number of unique nodes (senders and receivers).
    """
    # Step 1: Download Enron data
    data_url = "https://www.cis.jhu.edu/~parky/Enron/execs.email.linesnum.topic"
    response = requests.get(data_url)
    response.raise_for_status()
    raw_data_text = response.text
    raw_data_cleaned = raw_data_text.replace('"', '').strip()

    # Step 2: Parse valid lines into a DataFrame
    lines = raw_data_cleaned.split("\n")
    valid_lines = [line for line in lines if len(line.split()) == 5]
    filtered_data = "\n".join(valid_lines)
    filtered_data_io = StringIO(filtered_data)

    df = pd.read_csv(filtered_data_io, sep="\s+", names=["time", "sender", "receiver", "topic"])
    df["time"] = pd.to_datetime(df["time"], unit='s')
    df["time_group"] = df["time"].dt.to_period("W")

    # Step 3: Create binary adjacency matrices
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
        adjacency_data.append(adj_matrix)

    # Step 4: Convert to 3D numpy array
    adjacency_matrices = np.stack(adjacency_data, axis=2)
    return adjacency_matrices, num_nodes
