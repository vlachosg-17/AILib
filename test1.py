import numpy as np
# import matplotlib.pyplot as plt
# import argparse

from dl.model import Model
from dl.layers import MLP, ReLu, Softmax, Flatten, CNL2D
from utils.functions import *
from utils.dbs import DataBase

train_data, train_labels = DataBase("P:\data").load_digits("train_digits.csv")
testX = DataBase("P:\data").load_digits("test_digits.csv", lables_true=False)
train_labs = one_hot(train_labels)
X, y = shuffle(train_data, train_labs)
X = X.reshape(X.shape[0], 1, 28, 28)
testX = testX.reshape(testX.shape[0], 1, 28, 28)
X01, y, testX01 = to_01(X[:200]), y[:200], to_01(testX[:200])
# Neural Net's Architecture
layers = [
      CNL2D([(1, 28, 28), (100, 22, 22)])
    , ReLu()
    , Flatten()
    , MLP([100*22*22, 10])
    , Softmax() # end 2nd hidden or output layer
    ]
Net = Model(pipline=layers, loss="entropy")
# y_hat = Net(X01)
# print(y_hat.shape)
# print(y_hat)
print(Net.Layers[0].weights.shape, Net.Layers[0].bias.shape)
print(Net.Layers[3].weights.shape, Net.Layers[3].bias.shape)
# if hpars.train:
Net.train(X01, y, epochs=30, lr=0.005, batch_size=5, save_path=None)

y_prob = np.round(Net.prob(testX01), 3)
y_pred = Net.predict(testX01)


sample=np.random.choice(testX01.shape[0], 1)
# sample=range(0, 5)
for p, img in zip(y_prob[sample], testX01[sample]):
    f, (ax1, ax2) = plt.subplots(1, 2) 
    ax1.imshow(img.reshape(28, 28))
    ax1.set_title('The image')

    ax2.bar([i for i in range(10)], p)
    ax2.set_title('The prediction')
    ax2.set_xticks([i for i in range(10)])

    plt.tight_layout()
    plt.show()

n=5
sample=np.random.choice(testX01.shape[0], n * n)
testXs = testX01[sample]
y_preds = y_pred[sample]
fig, axs = plt.subplots(n, n)
for i in range(axs.shape[0]):
    for j in range(axs.shape[1]):
        axs[i, j].imshow(testXs[i+n*j].reshape(28, 28))
        axs[i, j].set_title("Pred: "+ str(y_preds[i+n*j]))
        axs[i, j].set_axis_off()
plt.tight_layout()
plt.show()


plt.plot([e for e in range(len(Net.train_errors))], Net.train_errors, label="train error")
plt.plot([e for e in range(len(Net.valid_errors))], Net.valid_errors, label="test error")
plt.legend(loc="upper right")
plt.show()
# testX01 = to_01(testX)
# y_prob = Net.prob(testX01)
# y_pred = Net.predict(testX01)AA
# y_test = testY
