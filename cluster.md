## clustering algo:

similarity

data reduction

method:

* based on time points(similarity in time): k-nn
* based on shape (similarity in space): dtw
* based on change (similarity in data generating): gmm, arma-mixture

feature:

* shape
* noise
* Translation
* amplitude

![1708963794746](image/cluster/1708963794746.png)

[8种时间序列分类方法总结 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/600353884)

[传统时间序列分类综述（单变量） - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/434581898)

[两篇关于时间序列的论文 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/50698719)

[时间序列聚类的2篇综述 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/432189783)


code:

[GitHub - benfulcher/hctsa: Highly comparative time-series analysis](https://github.com/benfulcher/hctsa)

[GitHub - hfawaz/dl-4-tsc: Deep Learning for Time Series Classification](https://github.com/hfawaz/dl-4-tsc)

[GitHub - timeseriesAI/tsai](https://github.com/timeseriesAI/tsai)

### 多元time series - TICC

[多元时间序列聚类：KDD2017 论文《Toeplitz Inverse Covariance-Based Clustering of Multivariate Time Series Data》精读 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/459773743)

[GitHub - davidhallac/TICC](https://github.com/davidhallac/TICC)

不知道为啥不行，threshold改到1e-2了但还是报错说不能有Null或者inf: '**LinAlgError**: Array must not contain infs or NaNs'

![1708974330390](image/cluster/1708974330390.png)

### DTC

[用深度学习做时序聚类：DTC模型 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/450653137#:~:text=%E7%8E%B0%E6%9C%89%E7%9A%84%E6%97%B6%E9%97%B4%E8%81%9A%E7%B1%BB%E6%96%B9%E6%B3%95%E7%9A%84%E7%A0%94%E7%A9%B6%E4%B8%BB%E8%A6%81%E9%9B%86%E4%B8%AD%E5%9C%A8%E8%A7%A3%E5%86%B3%20%E4%B8%A4%E4%B8%AA%E6%A0%B8%E5%BF%83%E9%97%AE%E9%A2%98%20%E4%B9%8B%E4%B8%80%3A%20%E6%9C%89%E6%95%88%E7%9A%84%E9%99%8D%E7%BB%B4%E5%92%8C%E9%80%89%E6%8B%A9%E5%90%88%E9%80%82%E7%9A%84%E7%9B%B8%E4%BC%BC%E6%80%A7%E5%BA%A6%E9%87%8F,%E3%80%82%20%E6%9C%AC%E6%96%87%E6%98%AF%E5%9B%B4%E7%BB%95%E8%BF%99%E4%B8%A4%E7%82%B9%E5%B1%95%E5%BC%80%E7%9A%84%E3%80%82%20%E8%AE%BA%E6%96%87%E8%83%8C%E6%99%AF%EF%BC%9A%20%E4%B8%8D%E5%90%8C%E4%BA%8E%E4%B8%80%E8%88%AC%E7%9A%84%E9%9D%99%E6%80%81%E6%95%B0%E6%8D%AE%EF%BC%8C%E6%97%B6%E5%BA%8F%E6%95%B0%E6%8D%AE%E5%86%85%E6%A0%B7%E6%9C%AC%E6%9F%90%E6%97%B6%E5%88%BB%E7%9A%84%E7%8A%B6%E6%80%81%E5%8F%98%E5%8C%96%E4%BC%9A%E4%B8%8E%E4%B9%8B%E5%89%8D%E5%90%8E%E6%97%B6%E5%88%BB%E7%8A%B6%E6%80%81%E7%9B%B8%E5%85%B3%EF%BC%8C%E4%BE%8B%E5%A6%82%E5%A4%A9%E6%B0%94%E6%95%B0%E6%8D%AE%E3%80%81%E8%AF%AD%E8%A8%80%E6%95%B0%E6%8D%AE%E7%AD%89%E7%AD%89%EF%BC%8C%E9%82%A3%E4%B9%88%E5%A6%82%E4%BD%95%E6%8D%95%E6%8D%89%E5%88%B0%E8%BF%99%E7%A7%8D%E5%8F%98%E5%8C%96%E8%A7%84%E5%BE%8B%E9%9D%9E%E5%B8%B8%E9%87%8D%E8%A6%81%EF%BC%8C%E5%BE%80%E5%BE%80%E9%9C%80%E8%A6%81%E7%BB%93%E5%90%88%E5%89%8D%E5%90%8E%E6%97%B6%E5%88%BB%E7%8A%B6%E6%80%81%E5%8E%BB%E5%88%86%E6%9E%90%E3%80%82%20%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%9C%A8%E6%9C%89%E6%A0%87%E7%AD%BE%E6%95%B0%E6%8D%AE%E9%9B%86%E4%B8%8A%E5%8F%96%E5%BE%97%E4%BA%86%E5%B7%A8%E5%A4%A7%E7%9A%84%E6%88%90%E5%8A%9F%EF%BC%8C%E7%9B%B8%E5%AF%B9%E8%80%8C%E8%A8%80%EF%BC%8C%E5%A4%8D%E6%9D%82%E3%80%81%E9%AB%98%E9%98%B6%E7%9A%84%E7%BB%93%E6%9E%84%E5%8C%96%E3%80%81%E5%A4%9A%E7%89%B9%E5%BE%81%E7%9A%84%E6%97%A0%E6%A0%87%E7%AD%BE%E6%95%B0%E6%8D%AE%E5%88%99%E8%8E%B7%E5%BE%97%E8%BE%83%E5%B0%91%E7%9A%84%E5%85%B3%E6%B3%A8%EF%BC%8C%E5%AF%B9%E4%BA%8E%E8%AF%A5%E7%B1%BB%E6%97%A0%E7%9B%91%E7%9D%A3%E6%95%B0%E6%8D%AE%EF%BC%8C%E4%BC%A0%E7%BB%9F%E7%9A%84%E5%81%9A%E6%B3%95%E6%98%AF%E5%88%A9%E7%94%A8%E6%9F%90%E7%A7%8D%E7%9B%B8%E4%BC%BC%E5%BA%A6%E8%A1%A1%E9%87%8F%E7%AE%97%E6%B3%95%E5%8E%BB%E8%BF%9B%E8%A1%8C%E8%81%9A%E7%B1%BB%EF%BC%8C%E4%BE%8B%E5%A6%82K-means%E3%80%81%E5%B1%82%E6%AC%A1%E8%81%9A%E7%B1%BB%E7%AD%89%E7%AD%89%EF%BC%8C%E4%BD%86%E6%98%AF%E8%BF%99%E4%BA%9B%E6%99%AE%E9%80%9A%E7%9A%84%E8%81%9A%E7%B1%BB%E7%AE%97%E6%B3%95%E5%8F%AA%E6%98%AF%E5%9C%A8%E9%9D%99%E6%80%81%E6%95%B0%E6%8D%AE%E4%B8%8A%E5%8F%96%E5%BE%97%E4%BA%86%E8%BE%83%E5%A5%BD%E7%9A%84%E6%95%88%E6%9E%9C%EF%BC%8C%E5%AF%B9%E4%BA%8E%E6%97%B6%E5%BA%8F%E6%95%B0%E6%8D%AE%E4%B8%8D%E6%98%AF%E5%BE%88%E9%80%82%E7%94%A8%E3%80%82)

又不行 狂错

![1708984034870](image/cluster/1708984034870.png)AttributeError: Exception encountered when calling TSClusteringLayer.call().

### TFC

[GitHub - mims-harvard/TFC-pretraining: Self-supervised contrastive learning for time series via time-frequency consistency](https://github.com/mims-harvard/TFC-pretraining)

### SOMTimes

运行成功，但是误差太大

![1709063065600](image/cluster/1709063065600.png)

哥们 20w 认真的？

[dtw_som/README.md at master · misilva73/dtw_som · GitHub](https://github.com/misilva73/dtw_som/blob/master/README.md)

[GitHub - JustGlowing/minisom: 🔴 MiniSom is a minimalistic implementation of the Self Organizing Maps](https://github.com/JustGlowing/minisom/tree/master)

### ROCKET

[超强超快的时间序列分类算法ROCKET：论文+代码详细解读 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/652509028)

[pyts.transformation.ROCKET — pyts 0.13.0 documentation](https://pyts.readthedocs.io/en/stable/generated/pyts.transformation.ROCKET.html)

### Statistical sklearn

用统计学特征然后聚类 方法包括: 

Agglomerative Clustering, kmeans, Bisecting KMeans

结果：

我不好说，我真不好说 

AgglomerativeClustering: 28 {'sil': 0.7484724278838539, 'cal': 559852.9872110004, 'dav': 0.21861121981666806}

kmeans: 21 {'sil': 0.7720657212977767, 'cal': 233297.67453092037, 'dav': 0.2144106652095423}

BiselectingKMeans: 20 {'sil': 0.7684913820683339, 'cal': 204026.9335473212, 'dav': 0.27674090408008634}

kernelKmeans: 很烂 14 {'sil': -0.006791893813434881, 'cal': 6.183761401276699, 'dav': 9.18906713239559}

Best model: BiKmeans Best number of clusters: 15 with score: 0.8581854385862734

![1709579000717](image/cluster/1709579000717.png)

k-shape: 不太行
