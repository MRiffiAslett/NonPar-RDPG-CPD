

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
