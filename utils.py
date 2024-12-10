import numpy as np
from scipy.stats import ks_2samp
from scipy.linalg import svd

def scaledPCA(A_mat, d):
    
    u, s, vt = svd(A_mat)
    xhat_mat = u[:, :d] @ np.diag(np.sqrt(s[:d]))
    return xhat_mat

def lowertri2mat(x, n, diag=False):
   
    M = np.zeros((n, n))
    idx = np.tril_indices(n, -1 if diag is False else 0)
    M[idx] = x
    if diag is False:
        M = M + M.T
    return M

def BICtype_obj(y, S):
    
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

    sorted_y = np.sort(y)

    for i in range(len(segments) - 1):
        s = segments[i]
        e = segments[i + 1]
        aux = y[s:e]
        aux = aux[~np.isnan(aux)]
        if len(aux) == 0:
            # if no data in this segment, continue
            continue

        aux_sorted = np.sort(aux)
        counts = np.searchsorted(aux_sorted, sorted_y, side="right")
        F_hat_z = counts / len(aux)

        mask = (F_hat_z > 0) & (F_hat_z < 1)
        F_hat_z = F_hat_z[mask]
        ny = len(F_hat_z)
        if ny == 0:
            continue

        val = np.sum(F_hat_z * np.log(F_hat_z) + (1 - F_hat_z) * np.log(1 - F_hat_z))
        cost += len(sorted_y) * len(aux) * val

    if obs_num > 0:
        cost = -cost + 0.2 * len(S) * (np.log(obs_num)) ** 2.1
    else:
        cost = np.inf

    return cost


def thresholdBS(BS_object, tau):
    
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
    
    if Parent is None:
        Parent = np.array([[s], [e]])

  
    intervals = np.vstack((Alpha, Beta))
    idx = np.where((intervals[0, :] >= s) & (intervals[1, :] <= e))[0]
    if len(idx) == 0:
        return {'S': np.array([]), 'Dval': np.array([]), 'Level': np.array([]), 'Parent': Parent}

    max_stat = 0
    best_cp = None
  
    for i in idx:
        A = intervals[0, i]
        B = intervals[1, i]
        candidate_points = range(A+delta, B-delta+1)
        for mid in candidate_points:
            stat = ks_2samp(Y_mat[:, A:mid].flatten(), Y_mat[:, mid:B].flatten())[0]
            if stat > max_stat:
                max_stat = stat
                best_cp = mid

    if best_cp is None:
        return {'S': np.array([]), 'Dval': np.array([]), 'Level': np.array([]), 'Parent': Parent}
    else:
        
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
   
    obs_num = data_mat.shape[1]

    if lowerdiag:
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
    xhat = np.array(xhat)  # shape (obs_num, n, d)

   
    Y_mat = np.zeros((n//2, obs_num))
    for t in range(obs_num):
        phat = xhat[t] @ xhat[t].T
        ind = np.random.permutation(n)
        for i in range(n//2):
            Y_mat[i, t] = phat[ind[2*i], ind[2*i+1]]

   
    N = np.full(obs_num, Y_mat.shape[0])
    BS_result = WBS_uni_nonpar(Y_mat, 0, obs_num, Alpha, Beta, N, delta)
    return BS_result

def tuneBSnonparRDPG(BS_object, data_mat, lowerdiag, d):
   
    obs_num = data_mat.shape[1]

    if lowerdiag:
       
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
        return None

    aux = np.sort(Dval)[::-1]  
    tau_grid = aux[:50]-1e-4
    tau_grid = tau_grid[~np.isnan(tau_grid)]
    tau_grid = np.concatenate((tau_grid, [10]))

    S_list = []
    for tau in tau_grid:
        cpt = thresholdBS(BS_object, tau)['cpt_hat']
        if cpt.size == 0:
            break
        cpt = np.sort(cpt)
        
        S_list.append(list(cpt))
   
    unique_S = []
    for s in S_list:
        if s not in unique_S:
            unique_S.append(s)
    score = []
    for S in unique_S:
        total_score = 0
        for i in range(Y_mat.shape[0]):
            # BICtype.obj takes a single row
            row_score = BICtype_obj(Y_mat[i, :], S)
            total_score += row_score
        score.append(total_score)

    no_cpt_score = 0
    for i in range(Y_mat.shape[0]):
        no_cpt_score += BICtype_obj(Y_mat[i, :], None)
    score.append(no_cpt_score)

    best_ind = np.argmin(score)
    if best_ind == len(score)-1:
        return None
    else:
        return unique_S[best_ind]

