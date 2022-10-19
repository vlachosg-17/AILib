from my_tree import MyDecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from utils.metric import *
from utils.helpers import split_sets, load_iris
import numpy as np
from tqdm import tqdm

def mytree_pred(trainX, trainY, teY, max_depth, min_size):
    tree=MyDecisionTreeClassifier(trainX, trainY, max_depth, min_size)
    return tree.predict(teY)

def sktree_pred(trainX, trainY, teY, max_depth, min_size):
    tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_size)
    tree.fit(trainX, trainY)
    return tree.predict(teY)

def evaluate_tree(epochs, data, labels, max_depth, min_size, my=True):
    acc, prec, rec, f1= [0]*epochs, [0]*epochs, [0]*epochs, [0]*epochs
    for i in tqdm(range(epochs), ncols=50, leave=False):
        data, labels = shuffle(data, labels)
        (trainX, trainY), (validX, validY) = split_sets(data, labels, n=round(data.shape[0]*(1-0.2)))
        if my:
            predY = mytree_pred(trainX, trainY, validX, max_depth, min_size)
        else:
            predY = sktree_pred(trainX, trainY, validX, max_depth, min_size)

        m = metrics(predY, validY)
        acc[i], prec[i], rec[i], f1[i] = m['accuracy'], m['precision'], m['recall'], m['Fmeasure']

    acc, prec, rec, f1 = \
        np.array(acc).reshape(-1,1), np.array(prec).reshape(-1,1),\
        np.array(rec).reshape(-1,1), np.array(f1).reshape(-1,1)
    np.savetxt("evaluate_tree_iris_max_depth_"+str(max_depth)+\
        "_min_size_"+str(min_size)+".csv", np.hstack((acc,prec,rec,f1)), delimiter=",")
    return {
        'mean_acc':np.mean(acc),
        'mean_prec':np.mean(prec),
        'mean_rec':np.mean(rec),
        'mean_F':np.mean(f1)
        }    

if __name__ == "__main__":
    data, labels = load_iris("data/iris.data", shuffle_data=True, seed=None)
    (trainX, trainY), (teX, teY) = split_sets(data, labels, n=round(data.shape[0]*(1-0.2)))
    print(evaluate_tree(10, trainX, trainY, 15, 40, my=True))
    print(evaluate_tree(10, trainX, trainY, 15, 40, my=False))