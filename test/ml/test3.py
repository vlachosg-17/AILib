from matplotlib import pyplot as plt

from utils.helpers import load_abalone, split_sets, over_sample
from graph_tree.plot_tree import dt_graph
from ml.tree import DTree
from ml.random_forest import RForest
from utils.metric import *

data, labels = load_abalone("data/abalone19.dat")
(trainX, trainY), (teX, teY) = split_sets(data, labels, test_ratio=0.5)
ostrainX, ostrainY = over_sample(trainX, trainY, k=(trainY=="negative").sum()-(trainY=="positive").sum(), l="positive")

tree = DTree(min_size=20, max_depth=15)
tree = tree.grow(ostrainX, ostrainY)
predY = tree.predict(teX)
m = metrics(predY, teY)
print_metrics(m)
# dt_graph(tree)
# plt.show()

rf = RForest(ntrees=50, min_size=1, max_depth = 30, n_vars=6)
rf = rf.fit(ostrainX, ostrainY)
predY = rf.predict(teX)
m = metrics(predY, teY)
print_metrics(m)

for n in range(len(rf.trees)):
    rf.trees[n].predict(teX)
rf.trees[5].predict(teX[2])

t = rf.trees[5]
print(t)
while True:
    if t.depth >= t.max_depth:
        break
    if len(t.children)!=0:
        for child in t.children:
            print((t.depth+1)*" " + f"{child.state} child: {t.children[1]}")
            t = child
    # if not t.leaf:
    #     t = t.right_child
    #     t = t.left_child


# while True:
if i == 0:
    print(t)
    print((t.depth+1)*" " + f"Right Child:{t.right_child}")
    print((t.depth+1)*" "  + f"Left Child:{t.left_child}")
    t = t.right_child
    t = t.left_child
else:
    # print(t)
    print((t.depth+1)*" " + f"Right Child:{t.right_child}")
    print((t.depth+1)*" "  + f"Left Child:{t.left_child}")

# rf.trees[1].predict(teX)
# rf.trees[2].predict(teX)



