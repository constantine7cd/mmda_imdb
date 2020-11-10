import numpy as np
import pandas as pd
import sklearn as sk

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
    from IPython.display import display

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
            print("Items:")
            items_in_claster = data_frame_clustered[item_id_label][data_frame_clustered['Cluster'] == claster_id]
            for item_id in items_in_claster:
                print(item_id)
        print()

# ------------------------ HW 3 ------------------------

def pivotal_conf_interval(feat_means, quantile=1.96, mean=None, std=None):
    if mean is None:
        mean = feat_means.mean()
        
    if std is None:
        std = feat_means.std()
    
    shift = quantile * std
    lbp_piv = mean - shift
    rbp_piv = mean + shift

    return lbp_piv, rbp_piv


def non_pivotal_conf_interval(feat_means, confidense=0.95):
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

