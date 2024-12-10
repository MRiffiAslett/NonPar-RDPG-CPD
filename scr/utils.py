import numpy as np
from scipy.stats import ks_2samp
from scipy.linalg import svd
import numpy as np
from scipy.stats import ks_2samp
from scipy.linalg import svd

def scaledPCA(A_mat, d):
    """
    Equivalent to scaledPCA in R.
    """
    u, s, vt = svd(A_mat)
    xhat_mat = u[:, :d] @ np.diag(np.sqrt(s[:d]))
    return xhat_mat

def lowertri2mat(x, n, diag=False):
    """
    Convert a vector of strictly lower-triangular elements of an n x n matrix into the full matrix.
    If diag=False, there is no diagonal in x; it only contains lower-triangular entries.
    """
    M = np.zeros((n, n))
    idx = np.tril_indices(n, -1 if diag is False else 0)
    M[idx] = x
    if diag is False:
        M = M + M.T
    return M

def BICtype_obj(y, S):
    """
    Compute BIC-type objective as in the R code BICtype.obj function.
    y is a 1D array (single row of Y_mat).
    S is a list of change points.
    """
    S = [] if S is None else S
    K = len(S)
    obs_num = len(y)

    # Remove NaNs
    y = y[~np.isnan(y)]

    # If no data left, return large cost
    if len(y) == 0:
        return np.inf

    cost = 0
    segments = [0] + list(map(int, S)) + [obs_num]  # Ensure indices are integers

    # Sort the full data once for ECDF calculations
    sorted_y = np.sort(y)

    # Iterate through segments
    for i in range(len(segments) - 1):
        s = segments[i]
        e = segments[i + 1]
        aux = y[s:e]
        aux = aux[~np.isnan(aux)]
        if len(aux) == 0:
            # if no data in this segment, continue
            continue

        # Compute ECDF on the entire sorted_y but only from aux
        aux_sorted = np.sort(aux)
        counts = np.searchsorted(aux_sorted, sorted_y, side="right")
        F_hat_z = counts / len(aux)

        # Remove cases where F_hat_z == 0 or 1
        mask = (F_hat_z > 0) & (F_hat_z < 1)
        F_hat_z = F_hat_z[mask]
        ny = len(F_hat_z)
        if ny == 0:
            continue

        # cost contribution:
        val = np.sum(F_hat_z * np.log(F_hat_z) + (1 - F_hat_z) * np.log(1 - F_hat_z))
        cost += len(sorted_y) * len(aux) * val

    # Add penalty term
    if obs_num > 0:
        cost = -cost + 0.2 * len(S) * (np.log(obs_num)) ** 2.1
    else:
        cost = np.inf

    return cost


def thresholdBS(BS_object, tau):
    """
    Threshold the BS_object at level tau.
    Returns a structure with cpt_hat.

    The BS_object has keys:
    - 'S': detected change points in order of detection
    - 'Dval': their associated CUSUM values
    - 'Level': levels at which each CP is found
    - 'Parent': intervals from which CPs are found

    thresholdBS filters these by Dval >= tau.
    """
    Dval = np.array(BS_object['Dval'])
    S = np.array(BS_object['S'])
    if len(S) == 0:
        return {'cpt_hat': np.array([])}
    # Change points that exceed the threshold tau
    idx = np.where(Dval >= tau)[0]
    if len(idx) == 0:
        return {'cpt_hat': np.array([])}

    return {'cpt_hat': S[idx]}

def WBS_uni_nonpar(Y_mat, s, e, Alpha, Beta, N, delta, level=1, Parent=None):
    """
    Univariate WBS for nonparametric RDPG, following logic of WBS.uni.nonpar in R.
    Y_mat: feature matrix (rows: features, cols: time)
    s, e: start and end indices of the current interval
    Alpha, Beta: arrays of random intervals
    N: array with dimension of Y_mat for each time index (used if needed)
    delta: minimum spacing

    Returns a dict like a "BS" object:
    { 'S': ..., 'Dval': ..., 'Level': ..., 'Parent': ... }
    """
    # Parent should track intervals
    if Parent is None:
        Parent = np.array([[s], [e]])

    # The R code picks intervals from Alpha,Beta and tries to find max KS statistic:
    # We'll simulate that logic:
    intervals = np.vstack((Alpha, Beta))
    # Filter intervals that lie inside [s,e]
    idx = np.where((intervals[0, :] >= s) & (intervals[1, :] <= e))[0]
    if len(idx) == 0:
        return {'S': np.array([]), 'Dval': np.array([]), 'Level': np.array([]), 'Parent': Parent}

    max_stat = 0
    best_cp = None
    # Compute KS statistic for each candidate split from these intervals
    # The R WBS code tries a binary segmentation approach:
    # For each interval (A,B), we find a point that maximizes KS statistic:
    for i in idx:
        A = intervals[0, i]
        B = intervals[1, i]
        # Check only splits with spacing > delta
        candidate_points = range(A+delta, B-delta+1)
        for mid in candidate_points:
            stat = ks_2samp(Y_mat[:, A:mid].flatten(), Y_mat[:, mid:B].flatten())[0]
            if stat > max_stat:
                max_stat = stat
                best_cp = mid

    if best_cp is None:
        # no change point found
        return {'S': np.array([]), 'Dval': np.array([]), 'Level': np.array([]), 'Parent': Parent}
    else:
        # Split the interval [s,e] into [s,best_cp] and [best_cp+1,e]
        # Then recurse on each subinterval
        left_res = WBS_uni_nonpar(Y_mat, s, best_cp, Alpha, Beta, N, delta, level+1, Parent)
        right_res = WBS_uni_nonpar(Y_mat, best_cp+1, e, Alpha, Beta, N, delta, level+1, Parent)

        # Combine results
        S = np.concatenate((left_res['S'], [best_cp], right_res['S']))
        Dval = np.concatenate((left_res['Dval'], [max_stat], right_res['Dval']))
        Level = np.concatenate((left_res['Level'], [level], right_res['Level']))
        # Parent keeps track of intervals, we can stack them if needed:
        # For simplicity, we keep only original parent intervals as done in R code:
        return {'S': S, 'Dval': Dval, 'Level': Level, 'Parent': Parent}

def WBS_nonpar_RDPG(data_mat, lowerdiag, d, Alpha, Beta, delta):
    """
    Corresponds to WBS.nonpar.RDPG in R.
    data_mat: numeric matrix of observations with time as columns, vectorized adjacency matrix as rows.
    lowerdiag: True if strictly lower-triangular, False if full adjacency.
    d: number of leading singular values for scaled PCA
    Alpha, Beta: random intervals for WBS
    delta: minimum spacing
    """
    obs_num = data_mat.shape[1]

    if lowerdiag:
        # Reconstruct full matrices from lower-triangular vector
        # In R: n = 1/2 + sqrt(2*nrow(data_mat)+1/4)
        n = int(0.5 + np.sqrt(2*data_mat.shape[0] + 0.25))
        full_data = np.zeros((n*n, obs_num))
        for i in range(obs_num):
            full_data[:, i] = lowertri2mat(data_mat[:, i], n, diag=False).flatten()
        data_mat = full_data
    else:
        n = int(np.sqrt(data_mat.shape[0]))

    # Perform scaledPCA (ASE) for each time point
    xhat = []
    for i in range(obs_num):
        A_mat = data_mat[:, i].reshape(n, n)
        xhat.append(scaledPCA(A_mat, d))
    xhat = np.array(xhat)  # shape (obs_num, n, d)

    # Construct Y_mat as in R code:
    # Y_mat = matrix(0, floor(n/2), obs_num)
    Y_mat = np.zeros((n//2, obs_num))
    for t in range(obs_num):
        phat = xhat[t] @ xhat[t].T
        ind = np.random.permutation(n)
        for i in range(n//2):
            Y_mat[i, t] = phat[ind[2*i], ind[2*i+1]]

    # Now perform WBS on Y_mat (univariate)
    # N = rep(nrow(Y_mat), obs_num) in R just means an array of shape obs_num with nrow(Y_mat)
    N = np.full(obs_num, Y_mat.shape[0])
    BS_result = WBS_uni_nonpar(Y_mat, 0, obs_num, Alpha, Beta, N, delta)
    return BS_result

def tuneBSnonparRDPG(BS_object, data_mat, lowerdiag, d):
    """
    Tune threshold for WBS results based on BIC-type scores as in R code.
    """
    obs_num = data_mat.shape[1]

    if lowerdiag:
        # reconstruct full adjacency matrices
        n = int(0.5 + np.sqrt(2*data_mat.shape[0] + 0.25))
        full_data = np.zeros((n*n, obs_num))
        for i in range(obs_num):
            full_data[:, i] = lowertri2mat(data_mat[:, i], n, diag=False).flatten()
        data_mat = full_data
    else:
        n = int(np.sqrt(data_mat.shape[0]))

    xhat = []
    for i in range(obs_num):
        A_mat = data_mat[:, i].reshape(n, n)
        xhat.append(scaledPCA(A_mat, d))
    xhat = np.array(xhat)

    Y_mat = np.zeros((n//2, obs_num))
    for t in range(obs_num):
        phat = xhat[t] @ xhat[t].T
        ind = np.random.permutation(n)
        for i in range(n//2):
            Y_mat[i, t] = phat[ind[2*i], ind[2*i+1]]

    Dval = BS_object['Dval']
    if len(Dval) == 0:
        # No change points found at all
        return None

    aux = np.sort(Dval)[::-1]  # decreasing order
    tau_grid = aux[:50]-1e-4
    tau_grid = tau_grid[~np.isnan(tau_grid)]
    tau_grid = np.concatenate((tau_grid, [10]))

    S_list = []
    for tau in tau_grid:
        cpt = thresholdBS(BS_object, tau)['cpt_hat']
        if cpt.size == 0:
            break
        cpt = np.sort(cpt)
        # Store as list of ints
        S_list.append(list(cpt))
    # Unique sets of S:
    # To ensure uniqueness, convert to tuples:
    unique_S = []
    for s in S_list:
        if s not in unique_S:
            unique_S.append(s)

    # Compute BICtype.obj for each set S and compare
    # plus the null model (no change points)
    score = []
    for S in unique_S:
        total_score = 0
        for i in range(Y_mat.shape[0]):
            # BICtype.obj takes a single row
            row_score = BICtype_obj(Y_mat[i, :], S)
            total_score += row_score
        score.append(total_score)

    # Add the no change point scenario
    no_cpt_score = 0
    for i in range(Y_mat.shape[0]):
        no_cpt_score += BICtype_obj(Y_mat[i, :], None)
    score.append(no_cpt_score)

    # Find the best scenario
    best_ind = np.argmin(score)
    if best_ind == len(score)-1:
        # no change points scenario is best
        return None
    else:
        return unique_S[best_ind]

