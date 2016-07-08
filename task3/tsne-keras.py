import numpy as np
from sklearn.manifold import TSNE
from keras.datasets import cifar10

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

# the data, shuffled and split between train and test sets
cifar = unpickle('cifar-10-batches-py/data_batch_1')
print(cifar.data)
#X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
#print(X.shape)
#model = TSNE(n_components=2, random_state=0)
#np.set_printoptions(suppress=True)
#model.fit_transform(X) 