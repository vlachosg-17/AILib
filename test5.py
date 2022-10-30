from utils.helpers import load_iris, split_sets
from utils.metric import *

from ml.naive_bayes import GNB

X, y = load_iris("data/iris.data", shuffle_data=True, seed=0)
(trX, trY), (teX, teY) = split_sets(X, y, test_ratio=0.5)
nb = GNB().fit(trX, trY) 
prY=nb(teX)
print(sum(prY == teY)/teY.shape[0])
m = metrics(prY, teY)
print_metrics(m)

