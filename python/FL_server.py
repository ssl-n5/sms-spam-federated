# MODEL AGGREGATOR

def FedAvg(clients,client_weights,server_model):
    import numpy as np
    import pandas as pd
    import re
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import (
        precision_score, recall_score, classification_report,
        accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay,  make_scorer
    )
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from sklearn import metrics
    import pickle
    from collections import Counter
    import warnings
    from sklearn.utils import resample
    import os
    from sklearn.utils import class_weight
    from sklearn.utils import shuffle

    max_num_sv = max([client.support_vectors_.shape[0] for client in clients])
    aggregated_support_vectors = np.zeros((max_num_sv, clients[0].support_vectors_.shape[1]))
    aggregated_labels = np.array([])

    # Aggregate the support vectors and labels based on client weights
    for client, weight in zip(clients, client_weights):
        num_sv = client.support_vectors_.shape[0]
        support_vectors = client.support_vectors_
        labels = client.predict(support_vectors)
        padding = max_num_sv - num_sv
        padded_support_vectors = np.pad(support_vectors, [(0, padding), (0, 0)], mode='constant')
        aggregated_support_vectors += weight * padded_support_vectors
        aggregated_labels = np.concatenate((aggregated_labels, labels))

    aggregated_labels = aggregated_labels[:max_num_sv]

    server_model.fit(aggregated_support_vectors, aggregated_labels)
    return server_model