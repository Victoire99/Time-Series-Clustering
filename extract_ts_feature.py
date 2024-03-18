
import numpy as np

from tsfresh.feature_extraction.feature_calculators import *
from sklearn.metrics import silhouette_score,calinski_harabasz_score,davies_bouldin_score
from sklearn.preprocessing import StandardScaler

from features import *
from load_and_process_data import *

import warnings
warnings.filterwarnings('ignore')

INPUT_PATH = "C:/WorkNote/data/"


def statClustering(model,file_list,main_dct):

    scaler = StandardScaler()
    
    #scaler = TimeSeriesScalerMeanVariance(mu=0.0, std=1.0)
    label_dct = {}
    scores_dct = {}

    for i in range(len(file_list)):

        name = file_list[i]

        arr = np.array(list(main_dct[name].values()))[1:]

        arr_normalized = scaler.fit_transform(arr)

        # time series statistical features
        features = np.zeros((arr.shape[0], 27))  

        for i, ts in enumerate(arr_normalized):
            features[i, 0] = np.mean(ts)
            features[i, 1] = np.std(ts)
            features[i, 2] = np.min(ts)
            features[i, 3] = np.max(ts)
            features[i, 4] = np.median(ts)
            features[i, 5] = np.percentile(ts, 25)
            features[i, 6] = np.percentile(ts, 50)
            features[i, 7] = np.percentile(ts, 75)
            features[i, 8] = np.percentile(ts, 90)
            features[i, 9] = sample_entropy(ts)
            features[i, 10] = distance(ts)
            features[i, 11] = entropy(ts)
            features[i, 12] = kurtosis(ts)
            features[i, 13] = skewness(ts)
            features[i, 14] = mean_abs_diff(ts)
            features[i, 15] = mean_diff(ts)
            features[i, 16] = median_abs_diff(ts)
            features[i, 17] = median_diff(ts)
            features[i, 18] = rms(ts)
            features[i, 19] = abs_energy(ts)
            features[i, 20] = binned_entropy(ts,10)
            features[i, 21] = c3(ts,2)
            features[i, 22] = cid_ce(ts,True)
            features[i, 23] = count_above_mean(ts)
            features[i, 24] = count_below_mean(ts)
            features[i, 25] = mean_second_derivative_central(ts)
            features[i, 26] = autocorr(ts)
       
        clusterer = model
        labels = clusterer.fit_predict(features)
        label_dct[name] = labels

        # close to 1 is better
        scores_dct['sil'] = silhouette_score(features, labels, metric='euclidean')
        scores_dct['cal'] = calinski_harabasz_score(features, labels)
        # close to 0 is better
        scores_dct['dav'] = davies_bouldin_score(features, labels)

    return label_dct, scores_dct

def replace_with_order(lst):
    mapping = {}
    new_lst = []
    counter = 0

    for num in lst:
        if num not in mapping:
            mapping[num] = counter
            counter += 1
        new_lst.append(mapping[num])

    return new_lst

