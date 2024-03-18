'''
Author: KEWEI ZHANG
Date: 2024-03-05 15:43:27
LastEditors: KEWEI ZHANG
LastEditTime: 2024-03-18 10:11:33
FilePath: \WorkNote\cluster\cluster_method.py
Description: 

'''
import pandas as pd
import numpy as np

import json
from minisom import MiniSom


from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering,KMeans,BisectingKMeans
from tslearn.clustering import TimeSeriesKMeans,KShape, KernelKMeans

from extract_ts_feature import *

import warnings
warnings.filterwarnings('ignore')

from tqdm.notebook import tqdm
import time

INPUT_PATH = "C:/WorkNote/data/"


    
#cus_dct,main_dct,file_list,custom_list = dctGenerator()


def statistical_cluster(file_list, main_dct):
    models_dct = {
        "Agglomerative Clustering":AgglomerativeClustering,
        "kmeans":KMeans,
        "BiKmeans":BisectingKMeans,
        "kshape":KShape,
        "KernelKMeans":KernelKMeans,
        "TimeSeriesKMeans":TimeSeriesKMeans
    }

    n_clusters_range = list(range(14,30))

    re_score_dct = {}
    re_label_dct = {}

    # progress bar
    total_iterations = len(models_dct) * len(n_clusters_range)
    progress_bar = tqdm(total=total_iterations)
    
    for model_name, model in models_dct.items():
        for n_clusters in n_clusters_range:
            model_instance = model(n_clusters=n_clusters)
            label_dct, scores_dct = statClustering(model_instance, file_list, main_dct)
            # replace the label with order
            for key in file_list:
                label_dct[key] = replace_with_order(label_dct[key])

            print(model_name, n_clusters, scores_dct)
            re_score_dct[model_name + str(n_clusters)] = scores_dct
            re_label_dct[model_name + str(n_clusters)] = label_dct
            progress_bar.update()

    with open('score_results.json', 'w') as f:
        json.dump(re_score_dct, f)
    with open('label_results.json', 'w') as f:
        json.dump(re_label_dct, f)
        
def som_cluster(cus_dct,file_list,custom_list,cluster_num=28,epoch=5000):
    id_list = []
    val_dct = {}
    for i in range(len(custom_list)):
        for j in range(len(file_list)):
            id = str(i)
            id_list.append(id)
            col_name = str(i) + '_' + str(j)
            val_dct[col_name] = cus_dct[custom_list[i]][file_list[j]]

    data = pd.DataFrame(val_dct)

    scaler = MinMaxScaler()
    #data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    data = scaler.fit_transform(data)

    # som method
    som_shape = (1, cluster_num)
    som = MiniSom(som_shape[0], som_shape[1], data.shape[1], sigma=0.8, learning_rate=.5,
                neighborhood_function='gaussian', random_seed=10)

    som.train_batch(data, epoch, verbose=True)

    # each neuron represents a cluster
    winner_coordinates = np.array([som.winner(x) for x in data]).T
    # with np.ravel_multi_index we convert the bidimensional
    # coordinates to a monodimensional index
    cluster_index = np.ravel_multi_index(winner_coordinates, som_shape)
    cluster_index = replace_with_order(cluster_index)

    return cluster_index