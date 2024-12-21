from RDPG import RDPG
from enron import load_enron_data
from SimulationStudy import SimulationStudy  
import numpy as np
import toml

def run_change_point_detection(adjacency_matrices, latent_dim, num_intervals, delta, threshold):
    rdpg = RDPG(latent_dim)
    num_timepoints = adjacency_matrices.shape[2]
    intervals = rdpg.generate_random_intervals(num_intervals, 0, num_timepoints)
    return rdpg.detect_change_points(adjacency_matrices, intervals, d=latent_dim, delta=delta, threshold=threshold)

dataset = "enron" 

if __name__ == "__main__":
    if dataset == "enron":
        config_path = "config/enron_config.toml"
    elif dataset == "simulation":
        config_path = "config/simulation_config.toml"
    
    config = toml.load(config_path)

    if dataset == "enron":
        adjacency_matrices = load_enron_data()
    elif dataset == "simulation":
        simulation = SimulationStudy()
        adjacency_matrices = simulation.generate_simulated_data(
            num_nodes=config["simulation"]["num_nodes"],
            latent_dim=config["simulation"]["latent_dim"],
            num_segments=config["simulation"]["num_segments"],
            num_samples_per_segment=config["simulation"]["num_samples_per_segment"]
        )

    change_points = run_change_point_detection(
        adjacency_matrices=adjacency_matrices,
        latent_dim=config["parameters"]["latent_dim"],
        num_intervals=config["parameters"]["num_intervals"],
        delta=config["parameters"]["delta"],
        threshold=None if config["parameters"]["threshold"] == 'none' else config["parameters"]["threshold"]
    )

    print(f"change points are {change_points[0]['S']}")
