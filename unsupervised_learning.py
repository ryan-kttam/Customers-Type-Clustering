import numpy as np
import pandas as pd


# load the data and prevent errors from occurring
try:
    data = pd.read_csv("customers.csv")
    data.drop(['Region', 'Channel'], axis=1, inplace=True)
    print "Wholesale customers dataset has {} samples with {} features each.".format(*data.shape)
except:
    print "Dataset could not be loaded. Is the dataset missing?"

# data summary (Mean, sd, percentile)
data.describe()

