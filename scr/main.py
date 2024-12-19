from RDPG import RDPG
from enron import load_enron_data
from SimulationStudy import SimulationStudy  
import numpy as np
import toml

def run_change_point_detection(adjacency_matrices, latent_dim, num_intervals, delta, threshold):
    """
    Runs the change point detection algorithm using the RDPG method.

    Args:
        adjacency_matrices (numpy.ndarray): 3D array of adjacency matrices.
        latent_dim (int): Latent dimension for RDPG.
        num_intervals (int): Number of random intervals to generate.
        delta (float): Detection sensitivity parameter.
        threshold (float): Detection threshold.

    Returns:
        List[int]: Detected change points.
    """
    rdpg = RDPG(latent_dim)
    num_timepoints = adjacency_matrices.shape[2]
    intervals = rdpg.generate_random_intervals(num_intervals, 0, num_timepoints)
    return rdpg.detect_change_points(adjacency_matrices, intervals, d=latent_dim, delta=delta, threshold=threshold)

# Choose dataset: "enron" or "simulation"
dataset = "enron"  # Change to "simulation" for simulation study

if __name__ == "__main__":
    # Load appropriate config file
    if dataset == "enron":
        config_path = "config/enron_config.toml"
    elif dataset == "simulation":
        config_path = "config/simulation_config.toml"
    else:
        raise ValueError("Invalid dataset name. Choose 'enron' or 'simulation'.")

    # Load configuration
    try:
        config = toml.load(config_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found at {config_path}")

    # Load and preprocess dataset
    if dataset == "enron":
        print("Loading Enron dataset...")
        adjacency_matrices, num_nodes = load_enron_data()
    elif dataset == "simulation":
        print("Loading Simulation data...")
        simulation = SimulationStudy()
        adjacency_matrices = simulation.generate_simulated_data(
            num_nodes=config["simulation"]["num_nodes"],
            latent_dim=config["simulation"]["latent_dim"],
            num_segments=config["simulation"]["num_segments"],
            num_samples_per_segment=config["simulation"]["num_samples_per_segment"]
        )

    # Run change point detection
    change_points = run_change_point_detection(
        adjacency_matrices=adjacency_matrices,
        latent_dim=config["parameters"]["latent_dim"],
        num_intervals=config["parameters"]["num_intervals"],
        delta=config["parameters"]["delta"],
        threshold=config["parameters"]["threshold"]
    )

    print("Detected change points:", change_points)
