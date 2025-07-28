import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


import sys
import logging
from functools import wraps

def log_to_file():
    """
    Decorator to log all print statements (including those from called functions) into a file.
    The log filename is determined by the last parameter (`fname`) of the decorated function.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract the last argument (assumed to be `fname`)
            if args:  # If positional arguments exist
                fname = args[-1]  
            else:  # If only keyword arguments exist, get the last kwarg
                fname = list(kwargs.values())[-1]  
            
            log_filename = f"{fname}_log.txt"  # Construct log filename

            # Set up logging
            logger = logging.getLogger(func.__name__)
            logger.setLevel(logging.INFO)
            file_handler = logging.FileHandler(log_filename, mode='a')
            formatter = logging.Formatter('%(asctime)s - %(message)s')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

            # Redirect stdout globally
            class PrintLogger:
                def write(self, message):
                    if message.strip():  # Avoid logging empty messages
                        logger.info(message.strip())

                def flush(self):
                    pass  # No need to flush anything

            original_stdout = sys.stdout  # Save original stdout
            sys.stdout = PrintLogger()  # Redirect stdout globally

            try:
                return func(*args, **kwargs)  # Execute the function and log everything
            finally:
                sys.stdout = original_stdout  # Restore original stdout
                logger.removeHandler(file_handler)  # Clean up handler

        return wrapper
    return decorator


def remove_feature_same_value_in_all_samples(df_data):
    to_drop = list(df_data.columns[df_data.nunique() == 1])
    print("  - ", len(to_drop), 'columns with the same value in all samples:')
    [print("\t", col) for col in to_drop]
    return df_data.drop(columns=to_drop, inplace=False)


def remove_low_variance_features(df_data, threshold=0.01):
    to_drop = list(df_data.columns[df_data.std() < threshold])
    print("  - ", len(to_drop), "columns with low variance:")
    [print("\t", col) for col in to_drop]
    return df_data.drop(columns=to_drop, inplace=False)


def combine_identical_features(df_data):
    # get clusters of genes that always present together in all the strains
    lst_feats = list(df_data.columns)
    lst_clusters = []
    while len(lst_feats) > 0 :
        feat = lst_feats[0]
        cluster = list([col for col in lst_feats if df_data.loc[:, col].equals(df_data.loc[:, feat])])
        lst_clusters.append(cluster)
        lst_feats = [feat for feat in lst_feats if feat not in cluster]
    print("  - ", len([cluster for cluster in lst_clusters if len(cluster) > 1]), "clusters of columns with identical values")
    [print('\t', cluster) for cluster in lst_clusters if len(cluster) > 1]
    df_data = df_data[[cluster[0] for cluster in lst_clusters]]
    df_data.columns = ['~'.join(cluster) for cluster in lst_clusters]
    return df_data
    

def combine_highly_correlated_features(df_data, threshold=1, viz_clusters=False):
    import networkx as nx
    corr_matrix = df_data.corr().abs()
    G = nx.Graph()
    G.add_nodes_from(corr_matrix.index)
    rows, cols = np.where(np.triu(np.abs(corr_matrix.values), k=1) > threshold)    # todo error! it's not the real subgroups
    for i, j in zip(rows, cols):
        G.add_edge(corr_matrix.index[i], corr_matrix.index[j])
    lst_clusters =  [set(component) for component in nx.connected_components(G)]
    print("  - ", len([cluster for cluster in lst_clusters if len(cluster) > 1]), "clusters of columns with correlation >=", threshold)
    [print('\t', cluster) for cluster in lst_clusters if len(cluster) > 1]
    print("  - ", len([cluster for cluster in lst_clusters if len(cluster) == 1]), 'columns not in clusters')
    if viz_clusters: 
        pos = nx.spring_layout(G)
        G.remove_nodes_from([node for node in G.nodes if G.degree(node) == 0])
        nx.draw(G, pos, with_labels=True, node_size=100, node_color='skyblue', font_size=6, font_color='gray', edge_color='gray')
        plt.title("Clusters in Correlation Graph")
        plt.show()
    df_data = pd.concat([df_data[cluster].mean(axis=1) for cluster in lst_clusters], axis=1)
    df_data.columns = ['~'.join(cluster) for cluster in lst_clusters]
    return df_data


# @log_to_file()    
def feature_reduction_pipeline(df_data, variance_threshold=0.01, correlation_threshold=0.8, viz_corr_clusters=False, fname=None):
    print('Original shape:', df_data.shape)
    df_data = remove_feature_same_value_in_all_samples(df_data)
    print('Shape after removing same value columns:', df_data.shape)
    df_data = remove_low_variance_features(df_data, variance_threshold)
    print('Shape after removing low variance columns:', df_data.shape)
    df_data = combine_identical_features(df_data)
    print('Shape after combining identical columns:', df_data.shape)
    df_data = combine_highly_correlated_features(df_data, correlation_threshold, viz_clusters=viz_corr_clusters)
    print('Shape after combining highly correlated columns:', df_data.shape)
    if fname is not None:
        df_data.to_csv(f'../data/{fname}.csv', sep='\t', header=True, index=True)
    
    return df_data