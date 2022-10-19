from matplotlib import pyplot as plt

from ml.random_forest import RForest
from graph_tree.plot_tree import dt_graph
from utils.helpers import load_iris, load_mushroom, split_sets
from utils.metric import *


data, labels = load_iris("data/iris.data", shuffle_data=True, seed=0)
(trainX, trainY), (teX, teY) = split_sets(data, labels, test_ratio=0.5)
rf = RForest(ntrees=250, min_size=10)
rf = rf.fit(trainX, trainY)
predY = rf.predict(teX)
m = metrics(predY, teY)
print_metrics(m)

# data, labels = load_mushroom("data/agaricus-lepiota.data")
# (trainX, trainY), (teX, teY) = split_sets(data, labels, test_ratio=0.2)
# rf = RForest(ntrees=55, min_size=10, max_depth = 30, n_vars=6)
# rf = rf.fit(trainX, trainY)
# # print(f"Total runtime: {(end_time - start_time)/60}")
# predY = rf.predict(teX)
# m = metrics(predY, teY)
# print_metrics(m)