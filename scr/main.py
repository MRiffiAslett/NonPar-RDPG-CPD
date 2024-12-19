from RDPG import RDPG
from enron import load_enron_data
import numpy as np

def run_change_point_detection(adjacency_matrices, latent_dim, num_intervals, delta, threshold):
    rdpg = RDPG(latent_dim)
    num_timepoints = adjacency_matrices.shape[2]
    intervals = rdpg.generate_random_intervals(num_intervals, 0, num_timepoints)
    return rdpg.detect_change_points(adjacency_matrices, intervals, d=latent_dim, delta=delta, threshold=threshold)

if __name__ == "__main__":
    print("Loading Enron dataset...")
    adjacency_matrices, num_nodes = load_enron_data()

    latent_dim = 5
    num_intervals = 30
    delta = 5
    threshold = None

    change_points = run_change_point_detection(
        adjacency_matrices=adjacency_matrices,
        latent_dim=latent_dim,
        num_intervals=num_intervals,
        delta=delta,
        threshold=threshold
    )

    print("Detected change points:", change_points)
