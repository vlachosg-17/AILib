import numpy as np
import pandas as pd
from functools import reduce

def load_mushroom(filePath):
    Data, Labels = [], []
    with open(filePath, "r") as f:
        for line in f:
            line = line.strip()
            if len(line)==0:
                break
            line = line.split(",")
            line = [l.strip() for l in line]
            Data.append(line[1:])
            Labels.append(line[0])
    data, labels = np.array(Data), np.array(Labels)
    return data, labels

class GNB:    
    def __call__(self, X):
        return self.predict(X)

    def pdf(self, X, mu, Sigma):
        def mnorm(x, mu, Sigma):
            return (2*np.pi)**(-x.shape[0]/2) \
                   * np.linalg.det(Sigma)**(-1/2) \
                   * np.exp(-(1/2) * np.dot((x-mu).T, np.linalg.inv(Sigma)).dot((x-mu)))
        if len(X.shape) < 2: X = X[:,np.newaxis]
        if len(mu.shape) < 2: mu = mu[np.newaxis]
        if len(Sigma.shape) < 3: Sigma = Sigma[np.newaxis]
        return np.array([[mnorm(x, m, s) for x in X] for m, s in zip(mu, Sigma)]).T

    def fit(self, X, y):
        self.classes, self.cnts = np.unique(y, return_counts=True)
        self.counts = dict(list(zip(self.classes, self.cnts)))
        self.I = [np.ix_(np.where(y==y_i)[0].tolist(), list(range(X.shape[1]))) for y_i in self.counts]
        self.mu = np.array([[np.mean(x) for x in X[I_k].T] for I_k in self.I])
        self.sigma = np.array([np.diag([np.std(x) for x in X[I_k].T]) for I_k in self.I])
        return self
        
    def predict(self, X):
        self.p_y = np.array([1/c for c in self.counts.values()])
        self.p_xy = self.pdf(X, self.mu, self.sigma)
        self.p  = self.p_y * self.p_xy
        self.y_hat = np.argmax(self.p, axis=1)
        return np.array([self.classes[p] for p in self.y_hat])

class MNB:    
    def __call__(self, X):
        return self.predict(X)

    def freq_per_level(self, x, label):
        # if is ducument it will make the unique words as levels
        dfrq = x.str.split(expand=True).stack().value_counts().to_frame()
        dfrq.columns = [f"C{label}"]
        # Create a column with unique levels of words
        dfrq.reset_index(inplace=True)
        return dfrq.rename(columns = {'index':'levels'})

    def freq_per_feature(self, x, I):
        # frequencies of each levels per class for this feature
        frqs = [self.freq_per_level(x.loc[I_k], k) for k, I_k in enumerate(I)]
        # Total vocabulary or number of levels of this feature (if feature contains documents then each 
        # each unique word of the document is a features too)
        vocab = pd.DataFrame(pd.concat(frqs).reset_index(drop=True).levels.unique(), columns=["levels"])
        # Inner Joins with vocab for max speed processing 
        fq = reduce(lambda  left,right: pd.merge(left, right, on=["levels"], how='left'), [vocab]+frqs)\
            .fillna(0) \
            .astype({f"C{l}": 'int32' for l in range(len(frqs))})
        return fq
    
    def prob_per_feature(self, freqs, a=1):
        probs = pd.DataFrame({"levels": freqs["levels"]})
        for i in range(len(self.counts)): 
            probs[f"C{i}"] = freqs[f"C{i}"].add(a).div(freqs[f"C{i}"].sum()+a)
        return probs

    def fit(self, X, y):
        self.classes, self.cnts = np.unique(y, return_counts=True)
        self.counts = dict(list(zip(self.classes, self.cnts)))
        self.I = [np.where(y==y_i)[0].tolist() for y_i in self.counts]
        self.freqs = [self.freq_per_feature(x, self.I) for _, x in X.iteritems()]
        self.probs = [self.prob_per_feature(fq) for fq in self.freqs]
        return self
        
    def predict(self, X):
        lb = [f"C{i}" for i in self.counts]
        self.p_y = np.log(np.array([1/c for c in self.counts.values()]))
        print(X.to_numpy())
        # self.p_xy = np.array([np.sum(x[lb].to_numpy() * np.log(p[lb].to_numpy()), axis=1) for x, p in zip(self.freqs, self.probs)])
        # self.p_xy = np.array([[np.prod([self.probs[i].loc[self.probs[i]["levels"]==X.iloc[0,i]][[f"C{k}"]] for i in range(X.iloc[0].shape[0])]) for k in range(len(self.counts))] for x in X])
        # np.array([[np.prod([self.probs[i].loc[self.probs[i]["levels"]==x[i]][[f"C{k}"]] for i in range(x.shape[0]) for k in range(len(self.counts))])] for x in X])
        self.p = np.exp(self.p_y + self.p_xy)
        self.y_hat = np.argmax(self.p, axis=1)
        return np.array([self.classes[p] for p in self.y_hat])

if __name__ == '__main__':
    # data = pd.read_csv("data\\train.csv")
    # X, y = pd.DataFrame(data.iloc[:6000, 1]), pd.DataFrame(data.iloc[:6000, 2])
    # nb = MNB().fit(X, y)
    # nb.predict(X)
    # print(nb.freqs[0][[f"C{i}" for i in nb.counts]].to_numpy())# .head(5))
    # print(nb.probs)
    # print(nb.p)
    # print(nb.counts)


    data, labels = load_mushroom("data/agaricus-lepiota.data")
    X, y = pd.DataFrame(data), pd.DataFrame(labels)
    nb = MNB().fit(X, y)
    print(nb.freqs[0][["C0"]])
    print(nb.probs)
    np.array([[np.prod([nb.probs[i].loc[nb.probs[i]["levels"]==X.iloc[0,i]][[f"C{k}"]].squeeze() for i in range(X.iloc[0].shape[0])]) for k in range(len(nb.counts))] for x in X.to_numpy()])





    print(X.iloc[0])
    Z = X.iloc[0].to_frame()
    Z.columns = ["levels"]
    pd.merge(Z, nb.probs[1].loc[nb.probs[1]["levels"]==Z.iloc[1,0]], on=["levels"], how='left')




















