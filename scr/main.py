from SimulationStudy import SimulationStudy
from RDPG import RDPG
import numpy as np

def run_change_point_detection(adjacency_matrices, latent_dim, num_intervals, delta, threshold):
    
    rdpg = RDPG(latent_dim)

    num_timepoints = adjacency_matrices.shape[2]
    intervals = rdpg.generate_random_intervals(num_intervals, 0, num_timepoints)

    change_points = rdpg.detect_change_points(
        adjacency_matrices=adjacency_matrices,
        intervals=intervals,
        d=latent_dim,
        delta=delta,
        threshold=threshold
    )
    return change_points
if __name__ == "__main__":
    num_nodes = 20
    latent_dim = 5
    num_segments = 3
    num_samples_per_segment = 50
    num_intervals = 30
    delta = 5
    threshold = None
    
    simulation = SimulationStudy()
    adjacency_matrices = simulation.generate_simulated_data(
        num_nodes=num_nodes,
        latent_dim=latent_dim,
        num_segments=num_segments,
        num_samples_per_segment=num_samples_per_segment
    )
    
    change_points = run_change_point_detection(
        adjacency_matrices=adjacency_matrices,
        latent_dim=latent_dim,
        num_intervals=num_intervals,
        delta=delta,
        threshold=threshold
    )
    
    print("Detected Change Points:", change_points)
    
