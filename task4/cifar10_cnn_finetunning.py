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

batch_size = 32
nb_classes = 10
nb_epoch = 200
data_augmentation = False

# input image dimensions
img_rows, img_cols = 32, 32
# the CIFAR10 images are RGB
img_channels = 3

weights_path="./temporal_weights/weights_cifar.hdf5"

# the data, shuffled and split between train and test sets
# (X_train2, y_train2), (X_test2, y_test2) = cifar10.load_data()

model = Sequential()

model.add(Convolution2D(32, 3, 3, border_mode='same',
                        input_shape=(img_channels, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# LOADING WEIGHTS TO FINE-TUNNE THEM
model.load_weights(weights_path)

model.layers.pop() 
model.layers.pop()

for layer in model.layers:
  layer.trainable= False

nb_classes=13

layer_last=Dense(nb_classes)
layer_last.trainable=True

layer_last2=Activation('softmax')
layer_last2.trainable=True

model.add(layer_last)
model.add(layer_last2)


# let's train the model using SGD + momentum (how original).
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])


(X_train, y_train), (X_test, y_test) = terrassa.load_data()

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

nb_classes = len(set(y_test))
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

checkpointer = ModelCheckpoint(filepath="./temporal_weights/weights_finetunned_terrassa.hdf5", verbose=1, save_best_only=True)

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
