from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.datasets import cifar10, mnist
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from utils.datasets import terrassa
from scipy.misc import imresize
from random import shuffle
import random

import numpy as np

nb_classes = 13
batch_size = 16
nb_epoch = 200
data_augmentation = False

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

def VGG_16(weights_path=None):
  model = Sequential()
  model.add(ZeroPadding2D((1, 1), input_shape=(3, 224, 224)))
  model.add(Convolution2D(64, 3, 3, activation='relu'))
  model.add(ZeroPadding2D((1, 1)))
  model.add(Convolution2D(64, 3, 3, activation='relu'))
  model.add(MaxPooling2D((2, 2), strides=(2, 2)))

  model.add(ZeroPadding2D((1, 1)))
  model.add(Convolution2D(128, 3, 3, activation='relu'))
  model.add(ZeroPadding2D((1, 1)))
  model.add(Convolution2D(128, 3, 3, activation='relu'))
  model.add(MaxPooling2D((2, 2), strides=(2, 2)))

  model.add(ZeroPadding2D((1, 1)))
  model.add(Convolution2D(256, 3, 3, activation='relu'))
  model.add(ZeroPadding2D((1, 1)))
  model.add(Convolution2D(256, 3, 3, activation='relu'))
  model.add(ZeroPadding2D((1, 1)))
  model.add(Convolution2D(256, 3, 3, activation='relu'))
  model.add(MaxPooling2D((2, 2), strides=(2, 2)))

  model.add(ZeroPadding2D((1, 1)))
  model.add(Convolution2D(512, 3, 3, activation='relu'))
  model.add(ZeroPadding2D((1, 1)))
  model.add(Convolution2D(512, 3, 3, activation='relu'))
  model.add(ZeroPadding2D((1, 1)))
  model.add(Convolution2D(512, 3, 3, activation='relu'))
  model.add(MaxPooling2D((2, 2), strides=(2, 2)))

  model.add(ZeroPadding2D((1, 1)))
  model.add(Convolution2D(512, 3, 3, activation='relu'))
  model.add(ZeroPadding2D((1, 1)))
  model.add(Convolution2D(512, 3, 3, activation='relu'))
  model.add(ZeroPadding2D((1, 1)))
  model.add(Convolution2D(512, 3, 3, activation='relu'))
  model.add(MaxPooling2D((2, 2), strides=(2, 2)))

  model.add(Flatten())
  model.add(Dense(4096, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(4096, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(1000, activation='softmax'))

  if weights_path:
    model.load_weights(weights_path)
  return model

(X_train, y_train), (X_test, y_test) = terrassa.load_data()

X_train2 = list()
y_train2 = list()
X_test2 = list()
y_test2 = list()


for i in range(len(X_train)):
  if(y_train[i] == 3): continue
  X_train2.append(X_train[i])
  y_train2.append(y_train[i])

r = list(range(len(X_test)))
random.shuffle(r)

j = 0
for i in r:
  j+=1
  if(y_test[i] == 3): continue
  if(j > 5000):
    X_train2.append(X_train[i])
    y_train2.append(y_train[i])
  else:
    X_test2.append(X_test[i])
    y_test2.append(y_test[i])
print("Remaining samples")
print("train: " + str(len(X_train2)))
print("test: " + str(len(X_test2)))


X_train = np.asarray(X_train2) 
y_train = np.asarray(y_train2)
X_test = np.asarray(X_test2)
y_test = np.asarray(y_test2)


print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

X_train2 = np.zeros([len(X_train), 3, 224, 224])
X_test2 = np.zeros([len(X_test), 3, 224, 224])

print(X_train[1, :, :, :].shape)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

for i in range(len(X_train)):
  if(y_train[i] == 3): continue
  image_aux = X_train[i, :, :, :]
  transposed_img = np.transpose(image_aux, (1, 2, 0))
  transposed_img[:, :, 0] -= 123.68
  transposed_img[:, :, 1] -= 116.779
  transposed_img[:, :, 2] -= 103.939
  image_aux2 = imresize(transposed_img, [224, 224])
  image_aux3 = np.transpose(image_aux2, (2, 0, 1))
  X_train2[i, :, :, :] = image_aux3

for i in range(len(X_test)):
  if(y_test[i] == 3): continue
  image_aux = X_test[i, :, :, :]
  transposed_img = np.transpose(image_aux, (1, 2, 0))
  transposed_img[:, :, 0] -= 123.68
  transposed_img[:, :, 1] -= 116.779
  transposed_img[:, :, 2] -= 103.939
  image_aux2 = imresize(transposed_img, [224, 224])
  image_aux3 = np.transpose(image_aux2, (2, 0, 1))
  X_test2[i, :, :, :] = image_aux3

X_train = X_train2
X_test = X_test2

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# X_train /= 255
# X_test /= 255


model = VGG_16("./weights/vgg16_weights.h5")
# model = convnet('VGG_16',weights_path="./weights/vgg16_weights.h5", heatmap=False)

pop_layer(model)

print len(model.layers)
for layer in model.layers:
  layer.trainable = False


layer_last = Dense(13, activation='softmax')
layer_last.trainable = True

model.add(layer_last)

sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='mse', metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath="./temporal_weights/weights_finetunned_alexnet_terrassa.hdf5", verbose=1,
                               save_best_only=True)

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


