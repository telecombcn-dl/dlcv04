'''Train a simple deep CNN on the CIFAR10 small images dataset.

GPU run command:
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cifar10_cnn.py

It gets down to 0.65 test logloss in 25 epochs, and down to 0.55 after 50 epochs.
(it's still underfitting at that point, though).

Note: the data was pickled with Python 2, and some encoding issues might prevent you
from loading it in Python 3. You might have to load it in Python 2,
save it in a different format, load it in Python 3 and repickle it.
'''

from __future__ import print_function
from keras.datasets import cifar10, mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from utils.datasets import terrassa
from scipy.misc import imresize
import numpy as np
import os
import random

batch_size = 64
nb_classes = 10
nb_epoch = 2000
data_augmentation = False

# input image dimensions
img_rows, img_cols = 32, 32
# the CIFAR10 images are RGB
img_channels = 3


def pop_layer(model):
    if not model.outputs:
        raise Exception('Sequential model cannot be popped: model is empty.')

    model.layers.pop()
    if not model.layers:
        model.outputs = []
        model.inbound_nodes = []
        model.outbound_nodes = []
    else:
        model.layers[-1].outbound_nodes = []
        model.outputs = [model.layers[-1].output]
    model.built = False



weights_path="./temporal_weights/weights_cifar.hdf5"

# the data, shuffled and split between train and test sets
# (X_train2, y_train2), (X_test2, y_test2) = cifar10.load_data()

model = Sequential()

model.add(Convolution2D(32, 3, 3, border_mode='same',
                        input_shape=(img_channels, img_rows, img_cols),trainable=False))
model.add(Activation('relu',trainable='False'))
model.add(Convolution2D(32, 3, 3,trainable='False'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), trainable='False'))
model.add(Dropout(0.25,trainable='False'))

model.add(Convolution2D(64, 3, 3, border_mode='same',trainable='False'))
model.add(Activation('relu',trainable='False'))
model.add(Convolution2D(64, 3, 3,trainable='False'))
model.add(Activation('relu',trainable='False'))
model.add(MaxPooling2D(pool_size=(2, 2),trainable='False'))
model.add(Dropout(0.25,trainable='False'))

model.add(Flatten(trainable='False'))
model.add(Dense(512,trainable='False'))
model.add(Activation('relu',trainable='False'))
model.add(Dropout(0.5,trainable='False'))
model.add(Dense(10,trainable='False'))
model.add(Activation('softmax',trainable='False'))

# LOADING WEIGHTS TO FINE-TUNNE THEM
model.load_weights(weights_path)

pop_layer(model)
pop_layer(model)

# for layer in model.layers:
#   layer.trainable= False

nb_classes=13

layer_last=Dense(nb_classes)
layer_last.trainable=True

layer_last2=Activation('softmax')
layer_last2.trainable=True

model.add(layer_last)
model.add(layer_last2)

print(model.summary())


# let's train the model using SGD + momentum (how original).
#sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer="sgd",
              metrics=['accuracy'])


(X_train, y_train), (X_test, y_test) = terrassa.load_data_without_unknown_class()

# reassign 40 samples to test (randomly), as sets are not well balanced
X_train2 = list(X_train)
y_train2 = list(y_train)
X_test2 = list()
y_test2 = list()

r = list(range(len(X_test)))
random.shuffle(r)

j = 0
for i in r:
  j+=1
  if(y_test[i] == 3): continue
  if(j > 40):
    X_train2.append(X_train[i])
    y_train2.append(y_train[i])
  else:
    X_test2.append(X_test[i])
    y_test2.append(y_test[i])


X_train = np.asarray(X_train2)
y_train = np.asarray(y_train2)
X_test = np.asarray(X_test2)
y_test = np.asarray(y_test2)

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

X_train2=np.zeros([len(X_train),3,32,32])
X_test2=np.zeros([len(X_test),3,32,32])

print(X_train[1,:,:,:].shape)

for i in range(len(X_train)):
    image_aux=X_train[i,:,:,:]
    transposed_img=np.transpose(image_aux, (1, 2, 0))
    image_aux2=imresize(transposed_img,[32,32])
    image_aux3=np.transpose(image_aux2, (2, 0, 1))
    X_train2[i,:,:,:]=image_aux3

for i in range(len(X_test)):
    image_aux=X_test[i,:,:,:]
    transposed_img=np.transpose(image_aux, (1, 2, 0))
    image_aux2=imresize(transposed_img,[32,32])
    image_aux3=np.transpose(image_aux2, (2, 0, 1))
    X_test2[i,:,:,:]=image_aux3

X_train=X_train2
X_test=X_test2

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

checkpointer = ModelCheckpoint(filepath="./temporal_weights/weights_cifar_finetunned_terrassa.hdf5", verbose=1, save_best_only=True)

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_data=(X_test, Y_test),
              shuffle=True,
              callbacks=[checkpointer])
else:
    print('Using real-time data augmentation.')

    # this will do preprocessing and realtime data augmentation
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(X_train)

    # fit the model on the batches generated by datagen.flow()
    model.fit_generator(datagen.flow(X_train, Y_train,
                        batch_size=batch_size),
                        samples_per_epoch=X_train.shape[0],
                        nb_epoch=nb_epoch,
                        validation_data=(X_test, Y_test),
                        callbacks=[checkpointer])
