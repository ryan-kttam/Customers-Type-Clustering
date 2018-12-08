# Customers-Type-Clustering
This project will analyze customers' annual spending amounts (in dollar) of different types of product categories. The goal of this project is to find the best variation (cluster) in the different types of customers.

*Details about the project is available [here](https://ryan-kttam.github.io/Customer_Type_Clustering/)*

## Getting Started
This project requires Python 2 and needs the following packages:
```
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
```

## Data Exploration
There are six product categories: 'Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', and 'Delicatessen'. To get a better understanding of the dataset visually, I constructed a scatter matrix of each of the six product features. I immediately noticed that most data points are concentrated at 0, and the density decreases as the spending amount increases. In addition, all features appear to be right skewed, meaning that majority of the data is on the lower values, and the mean is less than the median. This kind of skewness appears on Fresh vs Fresh, Milk vs Milk, etc. There are also many outliers, and those large values make it less obvious for us to see what exactly the distribution of the data are. If we somehow fixed those outliers, the distribution of the graphs might be look more like a normal distribution, at least not as extreme.
```
pd.scatter_matrix(data, alpha = 0.3, figsize = (14,8), diagonal = 'kde')
```

## Data Preprocessing
As observed, most categories are skewed to the right and contain outliers. I need to perform data transformation in order to neutralize the skewness in order to maximize the clustering algorithms performance.
To handle with right-skewed data, I apply natural logarithm to every data point:
```
log_data = np.log(data)
pd.scatter_matrix(log_data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');
```

To select outliers, I use tukey's method: a data point will be considered as an outlier when it is 1.5 times the interquartile range(IQR). That is, I will consider the data point is an outlier if the value is:
```
below (Q1) – (1.5 × IQR) or above (Q3) + (1.5 × IQR),
  where Q1 is 25% percentile, Q3 is 75% percentile, and IQR is Q3 - Q1
```

## Feature Transformation
I will be reducing the data dimensionality in this section. I will use PCA to draw clues about the underlying structure of the data. PCA calculates the dimensions that best maximize variance, meaning I will be able to find which compound combinations of categories best describe customers.
```
pca = PCA()
pca.fit(good_data)
```
Dimension 1 explained 47.7% of variance, while Dimension 2 explained 25.1% of variance. If I include the first four dimensions, I will have explained 93.52% variance. In order words, I can drop two dimensions and only lose about 6% of the variance. Even though including more dimensions usually means I will have more information, reducing the dimensionality of the data can significantly reduce the complexity of the problem. As a result, I chose to only include the first two dimensions in this case, which represent 72.8% of total explained variance.

## clustering
I will choose to use K-mean clustering algorithm to identify the various customer types hidden in the data. One of the advantages about K-mean clustering is that ecah data point is hard clustering, meaning it has to belong to one cluster. There are other clusterings algorithms, such as Gaussian Mixture Model clustering, where each data point is expressed in percentages, but in this particular case, I will be using K-mean clustering method.

### Determining the Number (K) of Clusters
While it is unclear that how many clusters best describe the data, we can quantify the fit of a clustering by calculating each data point's silhouette coefficient, which measures how similar a data point is to its assigned cluster: 1 indicating the most similar, and -1 indicating the least similar. I then calculate the mean of the silhouette coefficients, and select the cluster that has the highest mean to be the best K.

```
for n in range(2,10):
    # Kth cluster range from 2 to 9
    clusterer = KMeans(n_clusters = n)
    clusterer.fit(reduced_data)
    preds = clusterer.predict(reduced_data)
    score = silhouette_score(reduced_data, preds)
    print "For cluster = {}, the silhouette score is {}".format(n, score)
```

```
For cluster = 2, the silhouette score is 0.440604840149
For cluster = 3, the silhouette score is 0.345344449186
For cluster = 4, the silhouette score is 0.335711363131
For cluster = 5, the silhouette score is 0.359688146035
...
```
It turns out that the best K-th cluster is 2, and the centers (or means) of each segment definitely have a different distribution:

```
              Fresh	Milk	Grocery	Frozen	Detergents_Paper	Delicatessen
Segment 0	4380.0	8076.0	12328.0	926.0	4757.0	1130.0
Segment 1	9174.0	1922.0	2487.0	2049.0	318.0	697.0
```
Comparing Segment 0 and Segment 1, segment 1 has more than double for fresh and frozen products on average, while segments 0 spent significant amount on Milk, Grocery, Detergents_paper, and Delicatessen. We should be aware that segment 0 and segment 1 are just average customer spending that that specific segment. I would assume that segment 1 could represent restaurants, since it spent more money on Fresh and Frozen products, and segment 0 could represent supermarkets, or large grocery stores, because it spent a decent amount on every category, and its grocery is significantly higher than the mean as well as segment 1.

## Conclusion
Companies will often run A/B tests when making small changes to their products or services to determine whether making that change will affect its customers positively or negatively. In this particular case, the wholesale distributor can split the data into two groups (segment 0 and 1) and apply A/B testing to see whether one segment would perform a better result than the other. In fact, as I stated in the previous section, segment 1 is more likely to be restaurants and segment 0 is more likely to be supermarkets. It is possible that one segment prefers different type of change than the other. As a result, based on whether a particular segment react positively for this new change, the wholesale distributor could make a corresponding business decision.
