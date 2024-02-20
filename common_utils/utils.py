from sklearn import datasets
from common_utils import *
def get_samples_regression(num_samples=1000,num_features=2):
    return datasets.make_regression(n_samples=num_samples,n_features=num_features)


