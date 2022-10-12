import numpy as np
from PIL import Image
from matplotlib import image
from matplotlib import pyplot as plt
import pandas as pd
import utils.functions as F

with open("P:/data/cifar10/labels.txt") as f:
    print(f.name)
    classes = np.array([l.strip() for l in f])

img = Image.open("P:/data/cifar10/test/airplane/3_airplane.png")# .convert("L")
data = np.array(img)
# plt.imshow(data)
# plt.show()
# data = data.reshape([data.shape[2],data.shape[0],data.shape[1]])
# data.shape
# plt.imshow(data[0])

def to_01(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))

def to_0255(x):
    cuts = [[i * (1/255), (i+1) * (1/255)] for i in range(255)]
    dims = x.shape
    for ix, cut in enumerate(cuts):
        print(np.argwhere(x < cut[1]))
        break

def to_full_greyscale_mode(img, show = False):
    im = Image.fromarray(pd.DataFrame(np.array(img))\
                            .applymap(lambda x: 255 if x >= np.floor(255/2)  else 0)\
                            .to_numpy()\
                            .astype("uint8"))
    if show:
        im.show()
    return im

def rot180(X):
    Y = []
    N, M = X.shape
    i = N - 1
    while(i >= 0):
        j = M - 1
        while(j >= 0):
            Y.append(X[i, j])
            j = j - 1
        i = i - 1
    return np.array(Y).reshape(N, M)

class Conv2D:
    def __init__(self, d_in, padding=(0,0), stride=(1,1), **kwargs):
        """
        nfm: # of feature maps that will be created and thus future channels
        C: channels of starting image
        H: height
        W: width

        dims(input) = H x W x C
        dims(kernel) = nfm x n x m x C

        python start indexing from 0 then: 
        dims(output) = (H-n) x (W-m) x nfm

        esle if indexing start from 1 then:
        dims(output) = (H-n+1) x (W-m+1) x nfm
        """
        if "kernels" in kwargs.keys():
            self.k = kwargs["kernels"]
        if "b" in kwargs.keys():
            self.b = kwargs["b"]
        if "activation" in kwargs.keys():
            if kwargs["activation"] == "softmax":
                self.f = F.softmax
            if kwargs["activation"] == "relu":
                self.f = F.relu
            if kwargs["activation"] == "sigmoid":
                self.f = F.sigmoid
        else:
            self.f = F.relu
        self.id = "CNL"
        self.d_in = d_in
        self.pad_x = padding[0]
        self.pad_y = padding[1]
        self.stride_x = stride[0]
        self.stride_y = stride[1]
        self.nsteps_x = self.d_in[0]+ 2*self.pad_x - (self.k.shape[-3] - 1) - 1 + self.stride_x
        self.nsteps_y = self.d_in[1]+ 2*self.pad_y - (self.k.shape[-2] - 1) - 1 + self.stride_y
        self.d_out = [self.nsteps_x//self.stride_x, self.nsteps_y//self.stride_y, self.k.shape[0]]

        Herror=f"({self.d_in[0]}+2*{self.pad_x}-({self.k.shape[-3]}-1)-1+{self.stride_x})mod({self.stride_x}) = {self.nsteps_x}mod({self.stride_x}) = {self.nsteps_x%self.stride_x}"
        Werror=f"({self.d_in[1]}+2*{self.pad_y}-({self.k.shape[-2]}-1)-1+{self.stride_y})mod({self.stride_y}) = {self.nsteps_y}mod({self.stride_y}) = {self.nsteps_x%self.stride_x}"
        if self.nsteps_x%self.stride_x != 0:
            raise ValueError(Herror)

        if self.nsteps_y%self.stride_y != 0:
            raise ValueError(Werror)

        # self.num_params = np.prod(self.d_in) * np.prod(self.d_out) + 

    def __call__(self, X):
        return self.forward(X)

    def pad(self, X):
        return np.pad(X, [(self.pad_x, self.pad_x), (self.pad_y,self.pad_y), (0, 0)])
    
    def conv2d(self, X, filters):
        def ixs(start, dim):
            """ Provides ranges of lists e.g if start = 1, dim = 0 if kernel.shape[dim=0] = 3 then return range(1, 1+3) = [1,2,3] """
            return list(range(start, start + dim))
        def fm_blocks(kernels, xy_steps, xy_stride):
            nsteps_x, nsteps_y = xy_steps
            stride_x, stride_y = xy_stride
            k_x, k_y= kernels.shape[-3], kernels.shape[-2]
            W=range(0, nsteps_y, stride_y)
            H=range(0, nsteps_x, stride_x)
            return [(ixs(h, k_x), ixs(w, k_y)) for w in W for h in H]
        blocks = fm_blocks(filters, (self.nsteps_x, self.nsteps_y), (self.stride_x, self.stride_y))
        # X[np.ix_(xi, yi)].flatten().dot(k.flatten()) = np.sum(X[np.ix_(xi, yi)] * k)
        # Hz, Wz, Ck = self.d_out
        # A = X[n:Hz+n, m:Wz+m] slice of the matrix X
        # Q = np.array([np.dot(filters[p,n,m,:], X[n:Hz+n, m:Wz+m]) for n in range(self.d_out[0]) for m in range(self.d_out[1]) for p in range(self.d_out[2])])
        # in order to avoid python for loops with this method it removes 1 for loop
        return np.array([[np.sum(X[np.ix_(ix, iy)] * k) for (ix,iy) in blocks] for k in filters]).reshape(self.d_out)
        
    def forward(self, X):
        assert X.shape[0] == self.d_in[0] and X.shape[1] == self.d_in[1]
        self.x = X
        self.z = self.conv2d(self.pad(self.x), self.k)
        self.y = self.f(self.z)
        return self.y

    def backprop(self, node):
        self.DzL = 1
        pass

k1 = np.random.uniform(-np.sqrt(1/(3 * 7 * 7)), np.sqrt(1/(3 * 7 * 7)), size = [100, 7, 7, 3])
ConvLayer1 = Conv2D((32, 32), (1,1), (3,3), kernels=k1)
fms1 = ConvLayer1(data)
print("Starting image shape:", data.shape, "Feature map's shape:", fms1.shape)

k2 = np.random.uniform(-np.sqrt(1/(100 * 5 * 5)), np.sqrt(1/(100 * 5 * 5)), size = [200, 5, 5, 100])
ConvLayer2 = Conv2D(fms1.shape, (1,1), (1,1), kernels=k2)
fms2 = ConvLayer2(fms1)
print(f"Starting image shape:{fms1.shape}, Feature map's shape:{fms2.shape}, Number of parameters: {fms1.shape[-1]*np.prod(ConvLayer2.k.shape)}")

k3 = np.random.uniform(-np.sqrt(1/(200 * 4 * 4)), np.sqrt(1/(200 * 4 * 4)), size = [500, 4, 4, 200])
ConvLayer3 = Conv2D(fms2.shape, (0,0), (1,1), kernels=k3)
fms3 = ConvLayer3(fms2)
print(f"Starting image shape:{fms2.shape}, Feature map's shape:{fms3.shape}, Number of parameters: {fms2.shape[-1]*np.prod(ConvLayer3.k.shape)}")

k4 = np.random.uniform(-np.sqrt(1/(500 * 5 * 5)), np.sqrt(1/(500 * 5 * 5)), size = [1000, 5, 5, 500])
ConvLayer4 = Conv2D(fms3.shape, (0,0), (1,1), kernels=k4)
fms4 = ConvLayer4(fms3)
print(f"Starting image shape:{fms3.shape}, Feature map's shape:{fms4.shape}, Number of parameters: {fms3.shape[-1]*np.prod(ConvLayer4.k.shape)}")
