import numpy as np
import pandas as pd
import random


class Customers:

    def __init__(self):
        self.data = None

    def load_data(self, file):
        # load the data and prevent errors from occurring
        try:
            self.data = pd.read_csv(file)
            self.data.drop(['Region', 'Channel'], axis=1, inplace=True)
            print("Wholesale customers dataset has {} samples with {} features each.".format(*self.data.shape))
        except:
            print("Dataset could not be loaded. Is the dataset missing?")

    def get_random_sample(self, random_state = 0):
        # default random samples are [100, 200, 300]
        if random_state > 0:
            idx = random.sample(range(1,len(self.data)), 3)
        else:
            idx = [100, 200, 300]

        return self.data.iloc[idx].reset_index(drop=True)


customers_records = Customers()
customers_records.load_data('customers.csv')

# data summary (Mean, sd, percentile)
customers_records.data.describe()

samples = customers_records.get_random_sample()


# understanding the data
# Check whether column "Frozen" can be predicted by other columns
# That is: setting "Frozen" to be a response variable, while all other columns to be predictors
# I will use a decision tree regressor to perform such prediction
new_data = data.drop( ['Frozen'], axis=1)
label = data['Frozen']

# split the dataset into training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(new_data, label, train_size=0.75, random_state=1)

# Create a decision tree regressor and fit it to the training set
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(X_train,y_train)
# Report the score of the prediction using the testing set
score = regressor.score(X_test, y_test)
print score
# This score is R^2, which implies our features fails to fit and predict the "Frozen" category.
# In order word, the feature of "Frozen" is useful in identifying customers' spending habits.

# Visualizing the feature distributions
pd.scatter_matrix(data, alpha = 0.3, figsize = (14,8), diagonal = 'kde')

# Feature scaling
# apply transformation to the data (using natural logarithm to every columns)
log_data = np.log(data)
# apply natural logarithm to the samples
log_samples = np.log(samples)
pd.scatter_matrix(log_data, alpha = 0.3, figsize = (14,8), diagonal = 'kde')

# For each feature find the data points with extreme high or low values
for feature in log_data.keys():
    # Calculate Q1 (25th percentile of the data) for the given feature
    Q1 = np.percentile(log_data[feature], 25)
    # Calculate Q3 (75th percentile of the data) for the given feature
    Q3 = np.percentile(log_data[feature], 75)
    # Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
    step = (Q3 - Q1) * 1.5
    # Display the identified outliers
    print "Data points considered outliers for the feature '{}':".format(feature)
    # Note: "~" means not in array: e.g. ~np.array([True]) means False
    print log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))]

# indices to be removed
outliers = [66, 95, 96, 218, 338, 357, 86, 98, 154, 356, 75, 325, 161, 183, 109, 128, 142, 187, 233]
# Removing the outliers
good_data = log_data.drop(log_data.index[outliers]).reset_index(drop=True)

# Feature Transformation
# Apply PCA by fitting the good data with the same number of dimensions as features
from sklearn.decomposition import PCA
pca = PCA()
pca.fit(good_data)


# TODO: Transform log_samples using the PCA fit above
pca_samples = pca.transform(log_samples)

# show explained variances
print "Explained variances are shown from higheest to lowest (in percentage): {}, ".format(pca.explained_variance_ratio_*100)

#  Apply PCA by fitting the good data with only two dimensions
# Note: the explained variance will not change even when we change n to 2,
# the first dimension will still have 47.7% explained and second dimension still have 25.07% explained variance
pca = PCA(n_components=2)
pca.fit(good_data)
# Transform the good data using the PCA fit above
reduced_data = pca.transform(good_data)

# Transform log_samples using the PCA fit above
pca_samples = pca.transform(log_samples)

# Create a DataFrame for the reduced data
reduced_data = pd.DataFrame(reduced_data, columns = ['Dimension 1', 'Dimension 2'])


# creating clusters for the reduced data ( 2 dimensions)
# For this problem, the number of clusters is not known,
# so there is no guarantee that a given number of clusters is the best segment.
# However, by using "silhouette coefficient", we can quantify the goodness of a clustering.
# "silhouette coefficient" calculates how similar a data point is to its assigned cluster,
# -1 being disimilar and 1 being similar.
# Then the mean of the "silhouette coefficient" would be a scoring method to measure how well the clustering is.

# Apply a clustering algorithm to the reduced data
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# calculating the "silhouette coefficient" for number of clusters 2 to 9.
for n in range(2, 10):
    clusterer = KMeans(n_clusters = n)
    clusterer.fit(reduced_data)
    # Predict the cluster for each data point
    preds = clusterer.predict(reduced_data)
    # Calculate the mean silhouette coefficient for the number of clusters chosen
    score = silhouette_score(reduced_data, preds)
    print "For cluster = {}, the silhouette score is {}".format(n, score)

# Since cluster = 2 has the best silhouette score, I am selecting the number of clusters is 2.
clusterer = KMeans(n_clusters = 2)
clusterer.fit(reduced_data)
# Predict the cluster for each data point
preds = clusterer.predict(reduced_data)
# Find the cluster centers
centers = clusterer.cluster_centers_
# Predict the cluster for each transformed sample data point
sample_preds = clusterer.predict(pca_samples)


# transform the cluster centers to their real values
# Inverse transform the centers
log_centers = pca.inverse_transform(centers)
# Exponentiate the centers
true_centers = np.exp(log_centers)

# display the mean of the good data
print "The average of each category after removing outliers:"
print np.mean(np.exp(good_data))

# Display the true centers
segments = ['Segment {}'.format(i) for i in range(0,len(centers))]
true_centers = pd.DataFrame(np.round(true_centers), columns = data.keys())
true_centers.index = segments
print true_centers

# Cluster Analysis:
# Comparing Segment 0 and Segment 1, segment 1 has more than double for fresh and frozen products,
# while segments 0 is dominated on Milk, Grocery, Detergents_paper, and Delicatessen.
# We should be aware that segment 0 and segment 1 are just average customer spending that that specific segment.
# I would assume that segment 1 is more likely to represent restaurants, since it spent more money on Fresh
# and Frozen products. On the other hand, I would assume segment 0 to be supermarkets, or large grocery stores,
# because it has everything and specifically, the almost every categories are higher than the mean of our good data,
# and its grocery is significantly higher than the mean as well as segment 1.

# Given each sample point, predict which cluster it belongs to.
inv_tran_sample = pca.inverse_transform(pca_samples)
print pd.DataFrame(np.round(np.exp(inv_tran_sample)), columns = data.keys())
# Display the predictions
for i, pred in enumerate(sample_preds):
    print "Sample point", i, "predicted to be in Cluster", pred