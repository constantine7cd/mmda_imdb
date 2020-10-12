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

# ------------------------ HW 1 ------------------------

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

