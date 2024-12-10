import numpy as np


class SimulationStudy:
    def generate_simulated_data(self, num_nodes, latent_dim, num_segments, num_samples_per_segment, symm=True):
        adjacency_matrices = []

        for _ in range(num_segments):
            graphon_matrix = self.generate_graphon_matrix(num_nodes, latent_dim)

            for _ in range(num_samples_per_segment):
              
                adj_matrix = self.simulate_adjacency_from_graphon(graphon_matrix, symm=symm)
                adjacency_matrices.append(adj_matrix)

        return np.stack(adjacency_matrices, axis=2)

    @staticmethod
    def generate_graphon_matrix(num_nodes, latent_dim):
        latent_positions = np.random.rand(num_nodes, latent_dim) 
        graphon_matrix = latent_positions @ latent_positions.T
        graphon_matrix = np.clip(graphon_matrix, 0, 1)  
        np.fill_diagonal(graphon_matrix, 0)  
        return graphon_matrix

    @staticmethod
    def simulate_adjacency_from_graphon(graphon_matrix, symm=True):
       
        num_nodes = graphon_matrix.shape[0]

        if symm:
            adj_matrix = np.tril(np.random.binomial(1, graphon_matrix), k=-1)
            adj_matrix = adj_matrix + adj_matrix.T
        else:
            adj_matrix = np.random.binomial(1, graphon_matrix)

        return adj_matrix
