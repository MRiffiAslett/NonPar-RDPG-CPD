
class RDPG:
    def __init__(self, latent_dim: int):
        self.latent_dim = latent_dim

    @staticmethod
    def adjacency_spectral_embedding(adjacency_matrix, latent_dim):
        return scaledPCA(adjacency_matrix, latent_dim)

    def latent_position_alignment(self, latent_positions):
        aligned_positions = []
        base = latent_positions[:, :, 0]
        for t in range(latent_positions.shape[2]):
            x_t = latent_positions[:, :, t]
            u, _, vt = svd(x_t.T @ base)
            rotation = u @ vt
            aligned_positions.append(x_t @ rotation)
        return np.stack(aligned_positions, axis=2)

    @staticmethod
    def compute_ks_cusum_stat(data, s, t, e):
        segment_1 = data[:, s:t].flatten()
        segment_2 = data[:, t:e].flatten()
        segment_1 = segment_1[~np.isnan(segment_1)]
        segment_2 = segment_2[~np.isnan(segment_2)]
        if len(segment_1) == 0 or len(segment_2) == 0:
            return 0
        stat, _ = ks_2samp(segment_1, segment_2)
        return stat

    def detect_change_points(self, adjacency_matrices, intervals, d, delta, threshold=None):
        num_timepoints = adjacency_matrices.shape[2]
        num_nodes = adjacency_matrices.shape[0]

        embeddings = np.zeros((num_nodes, d, num_timepoints))
        for t in range(num_timepoints):
            embeddings[:, :, t] = self.adjacency_spectral_embedding(adjacency_matrices[:, :, t], d)

        aligned_positions = self.latent_position_alignment(embeddings)

        Y = np.zeros((num_nodes // 2, num_timepoints))
        for t in range(num_timepoints):
            phat = aligned_positions[:, :, t] @ aligned_positions[:, :, t].T

            ind = np.random.permutation(num_nodes)
            for i in range(num_nodes // 2):
                Y[i, t] = phat[ind[2*i], ind[2*i+1]]

        Alpha = np.array([interval[0] for interval in intervals])
        Beta = np.array([interval[1] for interval in intervals])
        N = np.full(Y.shape[1], Y.shape[0])

        BS_result = WBS_uni_nonpar(Y, 0, Y.shape[1], Alpha, Beta, N, delta)

        if threshold is None:
           
            data_mat = np.zeros((num_nodes*(num_nodes-1)//2, num_timepoints))

            lowerdiag = True
            if lowerdiag:
                # strictly lower tri entries:
                idx = np.tril_indices(num_nodes, -1)
                for tt in range(num_timepoints):
                    A = adjacency_matrices[:, :, tt]
                    data_mat[:, tt] = A[idx]
            else:
                data_mat = adjacency_matrices.reshape(num_nodes*num_nodes, num_timepoints)

            cpt_hat =  (BS_result, data_mat, lowerdiag, d)
            return cpt_hat
        else:
            cpt_hat = thresholdBS(BS_result, threshold)['cpt_hat']
            return cpt_hat

    @staticmethod
    def generate_random_intervals(num_intervals, lower, upper, min_spacing=5):
        """
        Generate random intervals for WBS.
        """
        intervals = []
        while len(intervals) < num_intervals:
            start = np.random.randint(lower, upper - min_spacing)
            end = np.random.randint(start + min_spacing, upper)
            intervals.append((start, end))
        return intervals
