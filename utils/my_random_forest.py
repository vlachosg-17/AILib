from utils.helpers import bootstrap_sample, load_iris, vote, split_sets, numerize_class
from my_tree import MyDecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from graph_tree.plot_tree import layout, TreeLayout
# from sklearn.metrics import confusion_matrix
from utils.metric import * 
import math as mt
import random
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

def oob_error(predictions, trainlabels):
    num_wrong_pred = 0
    for k in range(len(trainlabels)):
        if predictions[k] != trainlabels[k]:
            num_wrong_pred +=1
    
    return num_wrong_pred / len(trainlabels)    

class MyRandomForestClassifier:
    def __init__(self, X=None, y=None, ntrees=100, min_size=50, n_vars="auto", seed=None):
        self.type = "randomforest"
        self.ntrees=ntrees
        self.min_node_size = min_size
        self.n_vars = n_vars
        self.seed = seed
        self.X, self.y = X, y
        self.trees, self.data = [], []
        self.sktrees = []
        self.oob_score = None
        if X is not None and y is not None:
            self.fit(X, y)
            self.oob_score = self.out_of_bag(X, y)
        
    def subsamples(self, X, y, ratio=1.0):
        if self.seed is not None: np.random.seed(self.seed)
        for _ in range(self.ntrees):
            self.data.append(bootstrap_sample(X, y, int(len(X)*ratio)))
        return self.data

    def fit(self, X, y):
        self.X, self.y = X, y
        if self.n_vars is "auto":
            if X.shape[1]%2: m = int(mt.sqrt(X.shape[1]))
            else:            m = mt.floor(mt.log(X.shape[1])+1)
        elif self.n_vars is "all":
            m = self.X.shape[1]
        else:
            m = self.X.shape[1]
        samples = self.subsamples(X, y)
        for i, (Xnew, ynew) in enumerate(samples):
            # print(m, Xnew.shape, ynew.shape)
            self.trees.append(
                MyDecisionTreeClassifier(Xnew, ynew, node_size=self.min_node_size, max_features=m, label=i)
            )
            
    def predict(self, X):
        if len(self.trees) is 0: raise "fit data !!!"
        # for tree in self.trees:
        #     tr = TreeLayout().layout(tree.tree)
        #     ax = layout(tr)
        #     plt.show()
        trees_preds=[[tree.predict(x)[0] for tree in self.trees] for x in X]
        preds = [vote(trees_pred) for trees_pred in trees_preds]
        # for _, (t, p) in enumerate(zip(trees_preds, preds)):    
        #     print(t, vote(t), p)
        return preds

    def not_in_attrs(self, x):
        not_in_data = []
        for i, (data, _) in enumerate(self.data):
            if x in data:
                not_in_data.append(i)
        return not_in_data

    def out_of_bag(self, X, y):
        if len(self.trees) is 0: raise "fit data !!!"
        foob = []
        for x in X:
            trees_pred = []
            not_in = self.not_in_attrs(x)
            if len(not_in) != 0:
                for k in not_in:
                    trees_pred.append(self.trees[k].predict(x))
                foob.append(vote(trees_pred))
            else:
                foob.append(-999)

        return self.oob_error(y, foob)

    def oob_error(self, y_true, y_pred):
        num_wrong_pred = 0
        for k in range(len(y_true)):
            if y_pred[k] != y_true[k]:
                num_wrong_pred +=1
        
        return num_wrong_pred / len(y_true)   
    
def plot_oob_error(X, y, ntrees, min_size):
    error = []
    for t in tqdm(range(1, ntrees, 5)):
        rf = MyRandomForestClassifier(X, y, ntrees=t)
        error.append(rf.oob_score())
    plt.plot(np.arange(1,ntrees),error)
    plt.savefig('OUT-OF-BAG-ERROR-BREAST-CANCER.pdf')
    np.savetxt("oob_error_iris_50.csv", error, delimiter=",")
    return error


if __name__ == "__main__":
    num_tr, minbucket = 10, 10
    data, labels = load_iris("data/iris.data", shuffle_data=True, seed=0)
    (trainX, trainY), (teX, teY) = split_sets(data, labels, test_ratio=0.2)
    print(trainX.shape, trainY.shape, teX.shape, teY.shape)
    rf = MyRandomForestClassifier(trainX, trainY, num_tr, minbucket, n_vars="auto")
    print(rf.oob_score)
    mypredY = rf.predict(teX)
    m = metrics(teY, mypredY)
    print_metrics(m)
    
    # rf = RandomForestClassifier(n_estimators=num_tr, min_samples_split=minbucket, max_features=)
    # rf.fit(trainX, trainY)
    # # print(dir(rf))
    # skpredY = rf.predict(teX)
    # m = metrics(teY, skpredY)
    # print_metrics(m)
    # print(skpredY == mypredY)
    
    