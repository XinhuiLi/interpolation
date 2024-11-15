import glob
import mat73
import numpy as np
import scipy.io as sio
import scipy.stats as stats


def to_one_hot(x, m=None):
    if type(x) is not list:
        x = [x]
    if m is None:
        ml = []
        for xi in x:
            ml += [xi.max() + 1]
        m = max(ml)
    dtp = x[0].dtype
    xoh = []
    for i, xi in enumerate(x):
        xoh += [np.zeros((xi.size, int(m)), dtype=dtp)]
        xoh[i][np.arange(xi.size), xi.astype(int)] = 1
    return xoh


def load_sz_score(filename):
    """
    Load FBIRN subject measures

    :param filepath: filepath to the dataset
    :return valid_score: valid subject measures
    :return invalid_sub_ind: invalid subject indices
    """
    data_dict = mat73.loadmat(filename)
    keys = ['diagnosis(1:sz; 2:hc)','PANSS(positive)','PANSS(negative)',\
            'SpeedOfProcessing','AttentionVigilance','WorkingMemory','VerbalLearning',\
            'VisualLearning','ReasoningProblemSolving','CMINDS_composite', 'age', 'gender(1:male; 2:female)'] # PANSS: The Positive and Negative Syndrome Scale
    ind = [data_dict['FILE_ID'].index(i) for i in keys]
    score = data_dict['analysis_SCORE'][:,ind] # FBIRN:311x10

    n_sub = score.shape[0]
    for i in range(n_sub):
      if score[i,0]==2 and score[i,1]==-9999: #for positive PANSS, if CTR has -9999, assign minimal possible score 7
        score[i,1] = 7
      if score[i,0]==2 and score[i,2]==-9999: #for negative PANSS, if CTR has -9999, assign minimal possible score 7
        score[i,2] = 7

    # remove subjects with nan or -9999 entry
    invalid_score_ind = np.argwhere((np.isnan(score)) | (score==-9999))
    invalid_sub_ind = np.unique(invalid_score_ind[:,0])

    invalid_cognitive_score_ind = np.argwhere((np.isnan(score[:,3:])) | (score[:,3:]==-9999)) # remove subjects with cognitive score nan or -9999
    invalid_diagnosis_score_ind = np.argwhere((np.isnan(score[:,:3])) | (score[:,:3]==-9999)) # remove subjects with diagnosis score nan or SZ -9999
    invalid_score_ind = np.hstack([invalid_cognitive_score_ind[:,0], invalid_diagnosis_score_ind[:,0]])
    invalid_sub_ind = np.unique(invalid_score_ind)
    score_valid = np.delete(score, invalid_sub_ind, 0)

    return score_valid, invalid_sub_ind


def load_asd_score(filename):
    """
    Load ABIDE subject measures

    :param filepath: filepath to the dataset
    :return valid_score: valid subject measures
    :return site_valid: valid site labels
    :return invalid_sub_ind: invalid subject indices
    """
    data_dict = mat73.loadmat(filename)
    keys = ['DX_GROUP', 'ADOS_TOTAL', 'AGE_AT_SCAN', 'SEX'] # DX: 1 ASD, 2 CTR; ADOS: Autism Diagnostic Observation Schedule
    ind = [data_dict['FILE_ID'].index(i) for i in keys]

    site_ind = data_dict['FILE_ID'].index('SITE_ID')
    score_str_list = data_dict['analysis_SCORE_str']
    site_list = [score_str[site_ind] for score_str in score_str_list]

    score = data_dict['analysis_SCORE'][:,ind] # ABIDE1:869

    # set CTR ADOS score to 0
    missing_ados_ind = np.where( (score[:,0]==2) & (np.isnan(score[:,1])) )[0]
    score[missing_ados_ind, 1] = 0

    invalid_score_ind = np.argwhere(np.isnan(score))
    invalid_sub_ind = np.unique(invalid_score_ind[:,0])

    score_valid = np.delete(score, invalid_sub_ind, 0)
    site_valid = np.delete(site_list, invalid_sub_ind)

    return score_valid, site_valid, invalid_sub_ind


def load_sfnc(filename, nan_sub_ind=None):
    """
    Load sFNC data

    :param filepath: filepath to the dataset
    :param nan_sub_ind: invalid subject indices
    :return sfnc_triu: sFNC upper triangle data
    :return sfnc_raw: raw sFNC data
    """
    if 'ABIDE' in filename:
        data_dict = sio.loadmat(filename)
    else:
        data_dict = mat73.loadmat(filename)
    sfnc = data_dict['sFNC']
    sfnc_matrix_valid = np.delete(sfnc, nan_sub_ind, 0)

    # reshape sFNC
    sfnc_vector_valid = []
    for i in range(sfnc_matrix_valid.shape[0]):
      tmp = sfnc_matrix_valid[i]
      # only use the lower triangular part of the FNC (diagonal is all ones) and upper and lower triangular are mirrored
      tmp = tmp[np.triu_indices(53, 1)]
      sfnc_vector_valid.append(tmp)

    sfnc_vector_valid = np.array(sfnc_vector_valid)
    return sfnc_vector_valid, sfnc_matrix_valid


def load_dfnc(filepath, nan_sub_ind=None, dataset='FBIRN'):
    """
    Load dFNC data

    :param filepath: filepath to the dataset
    :param nan_sub_ind: invalid subject indices
    :return dfnc_tensor_valid: dFNC data
    """
    filelist = glob.glob(filepath)
    filelist.sort()
    dfnc_list = []
    if dataset.lower() == 'fbirn':
        for f in filelist:
            data_dict = sio.loadmat(f)
            dfnc = data_dict['FNCdyn']
            dfnc_list.append(dfnc)
    elif dataset.lower() == 'abide':
        for f in filelist:
            data_dict = mat73.loadmat(f)
            dfnc = data_dict['FNCdyn']
            dfnc_list.append(dfnc)
    dfnc_tensor = np.array(dfnc_list)
    dfnc_tensor_valid = np.delete(dfnc_tensor, nan_sub_ind, 0)
    return dfnc_tensor_valid


def vector2matrix(vector):
    """
    Convert a 1378x1 FNC vector to a 53x53 FNC matrix

    :param vector: 1378x1 FNC vector
    :return matrix: 53x53 FNC matrix
    """
    matrix = np.zeros((53, 53))
    matrix[np.triu_indices(53, 1)] = vector
    matrix[np.tril_indices(53, -1)] = matrix.T[np.tril_indices(53, -1)]
    matrix[np.diag_indices(53)] = 1
    return matrix


def compute_sub_per_state(kmeans_label, n_pt, n_state=5, n_window=137):
    """
    Compute the number of subjects per state

    :param kmeans_label: kmeans label
    :param n_pt: number of patients
    :param n_state: number of states
    :param n_window: number of windows
    :return num_sub_per_state: number of subjects per state
    :return ratio_sub_per_state: ratio of subjects per state
    """
    num_sub_per_state = np.zeros((2,n_state)) # 1st row: patient; 2nd row: control
    for i, j in enumerate(range(0,len(kmeans_label),n_window)):
        if i < n_pt:
            for k in range(5):
                if np.any(kmeans_label[j:j+n_window] == k):
                    num_sub_per_state[0,k] += 1
        else:
            for k in range(5):
                if np.any(kmeans_label[j:j+n_window] == k):
                    num_sub_per_state[1,k] += 1
    ratio_sub_per_state = num_sub_per_state / np.sum(num_sub_per_state, axis=0)
    return num_sub_per_state, ratio_sub_per_state


def compute_fnc_per_state(kmeans_label, n_pt, n_state=5, n_window=137):
    """
    Compute the number of FNCs per state

    :param kmeans_label: kmeans label
    :param n_pt: number of patients
    :param n_state: number of states
    :param n_window: number of windows
    :return num_fnc_per_state: number of FNCs per state
    :return ratio_fnc_per_state: ratio of FNCs per state
    """
    num_fnc_per_state = np.zeros((2,n_state)) # 1st row: patient; 2nd row: control
    for i in range(len(kmeans_label)):
        if i < n_pt * n_window:
            num_fnc_per_state[0,kmeans_label[i]] += 1
        else:
            num_fnc_per_state[1,kmeans_label[i]] += 1
    ratio_fnc_per_state = num_fnc_per_state / np.sum(num_fnc_per_state, axis=0)
    return num_fnc_per_state, ratio_fnc_per_state


def compute_dwell_state(kmeans_label_2d, sorted_state_ind, n_pt, n_hc, n_state=5):
    """
    Compute the occupancy rate per state

    :param kmeans_label_2d: kmeans label
    :param sorted_state_ind: sorted state indices
    :param n_pt: number of patients
    :param n_hc: number of controls
    :param n_state: number of states
    :return dwell_state_mean_pt: mean of occupancy rate for patients
    :return dwell_state_ste_pt: standard error of occupancy rate for patients
    :return dwell_state_mean_hc: mean of occupancy rate for controls
    :return dwell_state_ste_hc: standard error of occupancy rate for controls
    :return dwell_state_pvalue: p-value of t-test between patients and controls
    """
    n = kmeans_label_2d.shape[0]
    dwell_state = np.zeros((n, n_state))
    for i in range(n):
        for k in range(n_state):
            dwell_state[i,k]=len(np.where(kmeans_label_2d[i,:]==k)[0])
    dwell_state_sorted = dwell_state[:, sorted_state_ind]
    dwell_state_mean_pt = np.mean(dwell_state_sorted[:n_pt, :], axis=0)
    dwell_state_std_pt = np.std(dwell_state_sorted[:n_pt, :], axis=0)
    dwell_state_ste_pt = dwell_state_std_pt/np.sqrt(n_pt)
    dwell_state_mean_hc = np.mean(dwell_state_sorted[n_pt:, :], axis=0)
    dwell_state_std_hc = np.std(dwell_state_sorted[n_pt:, :], axis=0)
    dwell_state_ste_hc = dwell_state_std_hc/np.sqrt(n_hc)
    dwell_state_pvalue = np.zeros(n_state)
    for i in range(n_state):
        dwell_state_pt = dwell_state_sorted[:n_pt, i]
        dwell_state_hc = dwell_state_sorted[n_pt:, i]
        _, dwell_state_pvalue[i] = stats.ttest_ind(a=dwell_state_pt, b=dwell_state_hc)
    return dwell_state_mean_pt, dwell_state_ste_pt, dwell_state_mean_hc, dwell_state_ste_hc, dwell_state_pvalue


def compute_transition_matrix(kmeans_label_2d, sorted_state_ind, n_pt, n_state=5, n_window=137):
    """
    Compute the transition matrix

    :param kmeans_label_2d: kmeans label
    :param sorted_state_ind: sorted state indices
    :param n_pt: number of patients
    :param n_state: number of states
    :param n_window: number of windows
    :return transition_matrix_pt: transition matrix for patients
    :return transition_matrix_hc: transition matrix for controls
    :return transition_matrix: transition matrix
    :return transition_matrix_pvalue: p-value of t-test between patients and controls
    """
    n = kmeans_label_2d.shape[0]
    mapping = {}
    for k in range(n_state):
        mapping[k] = np.where(sorted_state_ind==k)[0][0]
    transition_matrix = np.zeros((n, n_state, n_state))
    for i in range(n):
        for t in range(n_window-1):
            state_t1 = mapping[kmeans_label_2d[i,t]]
            state_t2 = mapping[kmeans_label_2d[i,t+1]]
            if state_t1 != state_t2:
                transition_matrix[i, state_t1, state_t2] += 1
    transition_matrix_pt = np.mean(transition_matrix[:n_pt,:,:], axis=0)/n_window
    transition_matrix_hc = np.mean(transition_matrix[n_pt:,:,:], axis=0)/n_window
    
    transition_matrix_pvalue = np.zeros((n_state, n_state))
    for i in range(n_state):
        for j in range(n_state):
            _, transition_matrix_pvalue[i,j] = stats.ttest_ind(a=transition_matrix[:n_pt,i,j], b=transition_matrix[n_pt:,i,j])

    return transition_matrix_pt, transition_matrix_hc, transition_matrix, transition_matrix_pvalue


def find_unique_ind(sorted_state_ind, corr, ratio_fnc_per_state, n_state=5):
    """
    Find unique sorted state indices

    :param sorted_state_ind: sorted state indices
    :param corr: correlation matrix
    :param ratio_fnc_per_state: ratio of FNCs per state
    :param n_state: number of states
    :return unique_sorted_state_ind: unique sorted state indices
    """
    duplicated_dict = {}
    unique_ind = np.unique(sorted_state_ind)
    if len(sorted_state_ind)==len(unique_ind)+1:
        unique_sorted_state_ind = np.copy(sorted_state_ind)
        for i in unique_ind:
            ind = np.where(sorted_state_ind == i)[0]
            if len(ind) > 1:
                duplicated_dict[i] = ind
        missing_ind = list(set(np.arange(n_state)) - set(sorted_state_ind))
        duplicated_ind = list(duplicated_dict.keys())
        for i in duplicated_ind:
            duplicated_ind_loc = duplicated_dict[i]
            for j in missing_ind:
                ind = np.argmax(corr[0, duplicated_ind_loc, j])
                unique_sorted_state_ind[duplicated_ind_loc[ind]] = j
    elif len(sorted_state_ind)==len(unique_ind):
        unique_sorted_state_ind = sorted_state_ind
    else:
        unique_sorted_state_ind = np.argsort(ratio_fnc_per_state[1,:])
    return unique_sorted_state_ind