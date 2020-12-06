import numpy as np
import pandas as pd
import sklearn as sk

import numpy.linalg as L

# ------------------------ data preprocessing ------------------------

def quantification_scalar(data_frame, column_label):
    data_frame = data_frame.copy()
    unique_values = set(data_frame[column_label])
    for value in unique_values:
        label = f'{column_label}_{value}'
        data_frame[label] = (data_frame[column_label] == value).astype(int)
    return data_frame.drop(columns=[column_label])


def quantification_set(data_frame, column_label):
    data_frame = data_frame.copy()
    unique_values = set()
    for values in data_frame[column_label]:        
        unique_values |= set(values.split(','))

    for value in unique_values:
        label = f'{column_label}_{value}'
        data_frame[label] = len(data_frame)*[0]
        for i in range(len(data_frame)):
            data_frame[label][i] = 1 if value in data_frame[column_label][i] else 0
        
    return data_frame.drop(columns=[column_label])


def standardization(data_frame, columns_lebels=None, is_range_based=True):
    data_frame = data_frame.copy()
    if columns_lebels is None:
        columns_lebels = data_frame.columns

    for label in columns_lebels:
        column = data_frame[label]
        m =  column.mean()
        rng = column.max() - column.min()
        std = column.std()

        if is_range_based:
            data_frame[label] = (column - m) / rng
        else:
            data_frame[label] = (column - m) / std

    return data_frame

# ------------------------ HW 2 ------------------------

def cluster(data_frame, clusters_number, clustering_columns_labels=None):
    """
    Performs clustering of a object-feature data frame

    Args:
        data_frame (pd.DataFrame): matrix of shape [N, D], where
            N is a number of data points D - number of features
        clusters_number (int): number of clusters to compute
        clustering_columns_labels (list): array of labels of columns
            which will be used in clustering

    Returns:
        pd.DataFrame: copy of original data_frame with additional 
            column 'Cluster' which specifies index of cluster for 
            which particular data point corresponds to
        float: inertia value
    """

    import sklearn.cluster

    data_frame = data_frame.copy()
    if clustering_columns_labels is None:
        clustering_columns_labels = data_frame.labels

    data_frame_normalized = standardization(data_frame, clustering_columns_labels)
    clustering_data = data_frame_normalized[clustering_columns_labels].to_numpy()
    k_means = sk.cluster.KMeans(n_clusters=clusters_number, init='random', n_init=300, max_iter=10000, algorithm='full', random_state=42)
    result = k_means.fit(clustering_data)
    data_frame['Cluster'] = result.labels_

    return data_frame, result.inertia_


def print_clusters_info(data_frame_clustered, item_id_label=None, clustering_columns_labels=None):
    """
    Prints information about clusters in verbose and pretty manner (jupyter notebooks)

    Args:
        data_frame_clustered (pd.DataFrame): object-feature data frame which includes
            'Cluster' column in which specified index of a cluster of a data point

        item_id_label (str): key of a particular feature of a data set. Used to print
            5 data points of a feature which correspond to particular cluster

        clustering_columns_labels (list): array of features (str): keys of data frame 
            which were used for clustering

    Returns:
        None
    """

    from IPython.display import display
    from itertools import islice

    if clustering_columns_labels is None:
        clustering_columns_labels = data_frame_clustered.labels

    clusters_number = data_frame_clustered['Cluster'].max() + 1
    
    for claster_id in range(clusters_number):
        statistics = pd.DataFrame()
        
        for column_label in clustering_columns_labels:
            grand_mean = data_frame_clustered[column_label].mean()
            within_cluster_mean = data_frame_clustered[column_label][data_frame_clustered['Cluster'] == claster_id].mean()
            difference = within_cluster_mean - grand_mean
            relative_difference = difference / grand_mean

            statistics[column_label] = [within_cluster_mean, grand_mean, difference, relative_difference]
        statistics = statistics.sort_values(by=3, axis=1, key=lambda x: -abs(x))        
        statistics.insert(0, "", ["Within cluster mean", "Grand mean", 'Difference', 'Difference, %'])
        
        print(f'Cluster № {claster_id + 1}')
        print('------------------------------')
        display(statistics)
        if item_id_label is not None:
            items_in_claster = data_frame_clustered[item_id_label][data_frame_clustered['Cluster'] == claster_id]
            print(f'Cluster size: {len(items_in_claster)}')
            print()
            print(f'Few items from the cluster:')
            for item_id in islice(items_in_claster, 5):
                print('-', item_id)
        print()

# ------------------------ HW 3 ------------------------

def pivotal_conf_interval(feat_means, quantile=1.96, mean=None, std=None):
    """
    Computes pivotal confidense intervals for array
    of values

    Args:
        feat_means (np.ndarray): array of shape [D,]
        quantile (float): quantile value for standart gaussian distribution
            which specifies confidense for intervals
        mean (float): precomputed mean value of feat_means array. If None 
            specified mean value will be computed inside a function
        std (float): precomputed standart deviation value of feat_means array. 
            If None specified deviation value will be computed inside a function

    Returns:
        tuple: left and right values for pivotal confidense interval
    """
    if mean is None:
        mean = feat_means.mean()
        
    if std is None:
        std = feat_means.std()
    
    shift = quantile * std
    lbp_piv = mean - shift
    rbp_piv = mean + shift

    return lbp_piv, rbp_piv


def non_pivotal_conf_interval(feat_means, confidense=0.95):
    """
    Computes non-pivotal confidense intervals for array
    of values

    Args:
        feat_means (np.ndarray): array of shape [D,]
        confidense (float): confidense value for intervals. Has 
            a range from 0 to 1

    Returns:
        tuple: left and right values for non-pivotal confidense interval
    """

    n_trials = feat_means.shape[0]
    
    left = (1. - confidense) / 2.
    right = confidense + left

    left_idx = np.floor(left * n_trials).astype(np.int64)
    right_idx = np.floor(right * n_trials).astype(np.int64)

    means_sorted = np.sort(feat_means)
    lbp_non_piv = means_sorted[left_idx]
    rbp_non_piv = means_sorted[right_idx]

    return lbp_non_piv, rbp_non_piv


def bootstrap(
    feature, pivotal=True, non_pivotal=True, sample_size=None, \
    num_trials=5000, quantile=1.96, confidense=0.95):
    """
    Computes bootstrap for arbitary feature from data set. Bootstrap 
        may include pivotal and/or non-pivotal variations depending on
        input parameters

    Args:
        feature (np.ndarray): array of shape [N,], where N is a number
            of data points. Represents values of a particular feature for each
            data point
        pivotal (bool): specifies whether to compute pivotal bootstrap or not
        non_pivotal (bool): specifies whether to compute non-pivotal bootstrap or not
        sample_size (int): size of subset used in bootstrap each trial. If None is
            specified than sample size is set to N - number of data points
        num_trials (int): number of trials used in bootstrap
        quantile (float): if pivotal is true then quantile value should be specified.
            Default value is 1.96, used to compute 95% confidense intervals
        confidense (float): if non-pivotal is true then confidense value should be
            specified. Confidense has a range from 0 to 1. 

    Returns:
        dict: {
            'mean_trials' - np.ndarray of shape [num_trials, ], mean values of each trial;
            'mean': global mean;
            'std': global standart deviation;
        }

        if pivotal is True return value is tuple which in addition to dictionary includes
            tuple of left and right pivotal confidense boundaries  
        if non-pivotal is True return value is tuple which in addition to dictionary includes
            tuple of left and right non-pivotal confidense boundaries  
    """
    
    n = feature.shape[0]
    
    if sample_size is None:
        sample_size = n
    
    indices = np.random.uniform(
        low=0, high=n, size=(sample_size, num_trials))
    indices = np.floor(indices).astype(np.int64)
        
    feat_boots = feature[indices]
    feat_means = feat_boots.mean(axis=0)
    
    mean = feat_means.mean()
    std = feat_means.std()
    
    stats = {
        'mean_trials': feat_means,
        'mean': mean,
        'std': std
    }
    
    res = [stats]
    
    if pivotal:      
        conf_int_piv = pivotal_conf_interval(feat_means, quantile, mean, std)
        res.append(conf_int_piv)
        
    if non_pivotal:
        conf_int_non_piv = non_pivotal_conf_interval(feat_means, confidense)
        res.append(conf_int_non_piv)
    
    if len(res) > 1:
        return tuple(res)
    
    return stats

# ------------------------ HW 4 ------------------------

def contingency_table(df, first_feature, second_feature):
    # P(first_feature | second_feature)
    conditional_frequency_table = pd.crosstab(df[first_feature], df[second_feature], normalize='columns')

    quetelet_table = pd.crosstab(df[first_feature], df[second_feature])

    n = quetelet_table.to_numpy().sum()
    quetelet_table_copy = quetelet_table.copy()
    for i in range(len(quetelet_table)):
        for j in range(len(quetelet_table.columns)):
            pi = sum(quetelet_table_copy.iloc[i, :])
            pj = sum(quetelet_table_copy.iloc[:, j])
            quetelet_table.iloc[i, j] = n * quetelet_table.iloc[i, j] / (pi * pj) - 1

    return conditional_frequency_table, quetelet_table


def average_quetelet_index(conditional_frequency_table, quetelet_table):
    res = conditional_frequency_table.copy()
    res.iloc[:, :] = conditional_frequency_table.to_numpy() * conditional_frequency_table.to_numpy()
    return res


def numbers_of_observations_to_be_associated(average_quetelet_index, confidence_level):
    from scipy.stats import chi2
    from math import ceil

    n = len(average_quetelet_index)
    k = len(average_quetelet_index.columns)
    df = (n - 1)*(k - 1)

    x = average_quetelet_index.to_numpy().sum()
    
    return ceil(chi2.ppf(confidence_level, df) / x)

# ------------------------ HW 5 ------------------------

def zscoring_standartization(data, eps=1e-8):
    """
    Applies Z-scoring standartization: sutracts mean and divides by
        standart deviation

    Args: 
        data (np.ndarray): object-feature matrix of shape [N, D], where
            N - number of objects, D - feature dimensionality
        eps (float): small value used to avoid division by zero in the case
            when standart deviation = 0

    Returns:
        np.ndarray: object-feature matrix of shape [N, D]
    """

    return (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + eps)


def range_standartization(data, eps=1e-8):
    """
    Applies range standartization: sutracts mean and divides by
        range (max - min)

    Args: 
        data (np.ndarray): object-feature matrix of shape [N, D], where
            N - number of objects, D - feature dimensionality
        eps (float): small value used to avoid division by zero in the case
            when range = 0

    Returns:
        np.ndarray: object-feature matrix of shape [N, D]
    """
    range_ = np.max(data, axis=0) - np.min(data, axis=0)
    return (data - np.mean(data, axis=0)) / (range_ + eps)


def rank_standartization(data, eps=1e-8):
    """
    Applies rank standartization: sutracts min and divides by
        range (max - min)

    Args: 
        data (np.ndarray): object-feature matrix of shape [N, D], where
            N - number of objects, D - feature dimensionality
        eps (float): small value used to avoid division by zero in the case
            when range = 0

    Returns:
        np.ndarray: object-feature matrix of shape [N, D]
    """

    min_ = np.min(data, axis=0)
    range_ = np.max(data, axis=0) - min_
    return (data - min_) / (range_ + eps)


def data_scatter(data):
    """
    Computes Data scatter (sum of squares of the elements of a matrix)
        for object-feature data frame

    Args:
        data (np.ndarray): object-feature matrix of shape [N, D], where
            N - number of objects, D - feature dimensionality

    Returns:
        flow: data scatter scalar value of a data matrix
    """

    return np.sum(np.square(data)) 


def svd(data):
    """
    Computes singular value decomposition for arbitary object-feature
        data frame

    Args:
        data (np.ndarray): object-feature matrix of shape [N, D], where
            N - number of objects, D - feature dimensionality

    Returns:
        tuple: u, s, vh, where u - np.ndarray matrix of left singular vectors,
            s - np.ndarray vector of singular values, vh - np.ndarray matrix
            of right singular vectors. NOTE! For the purposes of homework u 
            and vh matrices are negated.
    """
    u, s, vh = L.svd(data, full_matrices=True)

    return -u, s, -vh


def svd_scatter_contrib(data):
    """
    Computes svd, data scatter, natural and per cent contributions of arbitary
        object-feature matrix. High-level routine which aggregates outputs of
        several functions for more convenient experiments

    Args:
        data (np.ndarray): object-feature matrix of shape [N, D], where
            N - number of objects, D - feature dimensionality

    Returns:
        tuple: u, s, vh - np.ndarrays outputs of singular-value decomposition,
            u and vh matrices are negated; scatter - float, Data scatter of
            data matrix; natural_contrib - np.ndarray array of shape [D,]
            which contains natural contributions for each feature; 
            percent_contrib - np.ndarray array of shape [D,] which contains 
            per cent contributions for each feature

    """
    scatter = data_scatter(data)
    u, s, vh = svd(data)

    natural_contrib = np.square(s)
    percent_contrib = natural_contrib / scatter

    return u, s, vh, scatter, natural_contrib, percent_contrib


def conventional_pca(X, standardized=False):    
    """
    Computes first two principal components of a matrix following 
        conventional PCA algorithm. Supported standartized and 
        non-standartized versions of data

    Args:
        X (np.ndarray): two dimensional array of shape [N, D], N - number
            of data points, D - dimensionality of each data point
        standartized (bool): identifies whether input data X is standartized 
            or not. 

    Returns: 
        tuple: two np.ndarray of shape [N,] - first and second principal
            components of data matrix
    """
    if standardized is False:
        mean_x = np.mean(X, axis=0)
        Y = np.subtract(X, mean_x) 
        B = (Y.T @ Y) / Y.shape[0] 
        L, C = np.linalg.eig(B)  
        sorted_idx = np.argsort(L)[::-1] 
        la1 = L[sorted_idx[0]]
        c1 = C[:, sorted_idx[0]]  
        pc1 = np.divide(Y @ c1, np.sqrt(Y.shape[0] * la1))    
        B_dot = B - la1 * np.multiply(c1, c1.T) 
        L_, C_ = np.linalg.eig(B_dot)
        argmax_ = np.argmax(L_)
        la2 = L_[argmax_]
        c2 = C_[:, argmax_]
        pc2 = np.divide(Y@c2, np.sqrt(Y.shape[0]*la2))  
    else:
        Y = X
        B = (Y.T @ Y) / Y.shape[0] 
        L, C = np.linalg.eig(B)
        sorted_idx = np.argsort(L)[::-1]  
        la1 = L[sorted_idx[0]]
        c1 = C[:, sorted_idx[0]]  
        pc1 = np.divide(Y @ c1, np.sqrt(Y.shape[0] * la1))  
        la2 = L[sorted_idx[1]]
        c2 = -C[:, sorted_idx[1]]  
        pc2 = np.divide(Y @ c2, np.sqrt(Y.shape[0] * la2))
    
    return pc1, pc2


def hidden_factor(data, z, c, mu, argmax):    
    """
    Computes loadings, score vectors, ranking factors and contribution
        for object-feature data

    Args:
        data (np.ndarray): two dimensional array of shape [N, D], N - number
            of data points, D - dimensionality of each data point
        z (np.ndarray): matrix of negated left singular vectors of a matrix data 
        c (np.ndarray): matrix of negated right singular vectors of a matrix data
        mu (np.ndarray): vectors of singular values of a matrix data
        argmax (int): index of highest singular value in vector mu (np.argmax(mu))
    """

    c1 = c[argmax, :]  
    alpha = 1 / np.sum(c1)
    score_vector = c1 * alpha  

    ranking_factors = 100 * data @ score_vector 
    contribution = mu[argmax] ** 2 / np.sum(np.square(data))

    return c1, score_vector, ranking_factors, contribution
