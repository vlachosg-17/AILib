import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

from dl.model import Model
from dl.layers import MLP, ReLu, Softmax
from utils.functions import *
from utils.dbs import DataBase

parser = argparse.ArgumentParser()
parser.add_argument("--main_path",              type=str,   default="H:\My Drive\ML\DLib")
parser.add_argument("--data_path",              type=str,   default="P:\data")
parser.add_argument("--train",                  type=bool,  default=False)
parser.add_argument("--epochs",                 type=int,   default=200)
parser.add_argument("--lr",                     type=float, default=0.003)
parser.add_argument("--batch_size",             type=int,   default=4)
# parser.add_argument("--hidden_layer_neurons",   type=int,   default=[100])
parser.add_argument("--test_prc",               type=int,   default=0.4)
parser.add_argument("--save_dir",         type=str,   default="pars\iris")
parser.add_argument("--latest_checkpoint_path", type=str,   default=None)
hpars = parser.parse_args()


def model(X, hpars):
    # Neural Net's Architecture
    layers = [
        MLP([X.shape[1], 100]), 
        ReLu(), # end of 1st hidden layer
        MLP([100, y.shape[1]]), 
        Softmax() # end 2nd hidden or output layer
        ]
    Net = Model(pipline=layers, loss="entropy", stored_path=f"{hpars.main_path}\{hpars.save_dir}")
    if hpars.train:
        Net.train(
            X=X, 
            y=y, 
            epochs=hpars.epochs, 
            lr=hpars.lr,
            batch_size=hpars.batch_size, 
            save_path=f"{hpars.main_path}\{hpars.save_dir}"
        )

    y_prob = Net.prob(testX)
    y_pred = Net.predict(testX)
    return {"probability": y_prob, "prediction": y_pred, "net": Net}
    
def tables(y_test, y_pred, y_prob):
    cm = confmtx(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    # print(y_test)
    # print(np.round(y_prob[:, 1], 3))
    print("Accuracy:", np.diag(cm.to_numpy()).sum()/cm.to_numpy().sum())
    print("AUC:", roc_auc_score(y_test, y_prob, multi_class="ovr"))
    print("AUC:", roc_auc_score(y_test, y_prob, multi_class="ovo"))

def make_plots(X=None, Net=None):
    if X is None and Net is None:
        exit()
    
    ### Plot Train and Test Error
    fig, ax = plt.subplots(figsize=(9,6))
    ax.plot([e for e in range(len(Net.train_errors))], Net.train_errors, label="Train Error")
    ax.plot([e for e in range(len(Net.valid_errors))], Net.valid_errors, label="Validation Error")
    ax.legend(loc="upper right")
    fig.tight_layout()
    plt.close()

    Y = pd.DataFrame(X, columns=["sepal_length", "sepal_width", "petal_legth", "petal_width", "classes", "pred"])\
          .astype({"classes":"int64", "pred":"int64"})\
          .assign(classes = lambda X: X["classes"].map({0: "Iris-Setosa", 1: "Iris-Versicolour", 2: "Iris-Virginica"}))\
          .assign(pred = lambda X: X["pred"].map({0: "Iris-Setosa", 1: "Iris-Versicolour", 2: "Iris-Virginica"}))\
          .assign(success = lambda X: (X.classes == X.pred))


    fig, ax = plt.subplots(figsize=(9,6))
    df = Y.groupby(["classes", "success"])["pred"].count()\
          .unstack(fill_value=0).stack().sort_index(level=[0,1])\
          .rename("counts").to_frame()\
          .join(Y.groupby(["classes"])["success"].count().rename("total_counts"), how="left", on=["classes"])\
          .assign(acc = lambda X: X["counts"] / X["total_counts"])\
          .assign(err = lambda X: 1 / X["total_counts"] * np.sqrt(X["counts"]*(1-X["counts"] / X["total_counts"])))\
          .reset_index()\
          .loc[lambda X: X.success==1]
    sns.barplot(data = df, x="classes", y="acc")
    ax.errorbar(x=ax.get_xticks(), y=df.acc, yerr=df.err, ls="", color="black", capsize=4)
    plt.show()

    fig, ax = plt.subplots(2, 3, figsize=(14, 12))
    sns.scatterplot(data=Y, x="sepal_length", y="sepal_width", style="success", hue="classes", markers={False: "X", True: "o"}, ax=ax[0,0])
    sns.scatterplot(data=Y, x="sepal_length", y="petal_legth", style="success", hue="classes", markers={False: "X", True: "o"}, ax=ax[0,1])
    sns.scatterplot(data=Y, x="sepal_length", y="petal_width", style="success", hue="classes", markers={False: "X", True: "o"}, ax=ax[0,2])
    sns.scatterplot(data=Y, x="sepal_width", y="petal_legth", style="success", hue="classes", markers={False: "X", True: "o"}, ax=ax[1,0])
    sns.scatterplot(data=Y, x="sepal_width", y="petal_width", style="success", hue="classes", markers={False: "X", True: "o"}, ax=ax[1,1])
    sns.scatterplot(data=Y, x="petal_legth", y="petal_width", style="success", hue="classes", markers={False: "X", True: "o"}, ax=ax[1,2])
    handles, labels = ax[0, 0].get_legend_handles_labels()
    for axs in ax.flatten():
        axs.get_legend().remove()
    fig.legend(handles=handles, labels=labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    plt.show()

    # print(Y)



if "__main__" == __name__:
    ### Iris Data Set
    data, labels = DataBase(hpars.data_path).load_iris("iris.data", lab_nom=True, random_seed=10000)
    labs = one_hot(labels)
    X, y, testX, testY = split(data, labs, hpars.test_prc)

    results = model(testX, hpars)
    Net = results["net"]
    y_prob = results["probability"] # Net.prob(testX)
    y_pred = results["prediction"] # Net.predict(testX)
    y_test = reverse_one_hot(testY, classes=np.unique(labels))

    tables(y_test, y_pred, y_prob)

    X = np.hstack([testX, y_test.reshape(-1,1), y_pred.reshape(-1,1)])

    make_plots(X=X, Net=Net)