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
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils

batch_size = 32
nb_classes = 10
nb_epoch = 10
data_augmentation = True

# input image dimensions
img_rows, img_cols = 32, 32
img_width = img_rows
img_height = img_cols
# the CIFAR10 images are RGB
img_channels = 3

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Convolution2D(32, 3, 3, border_mode='same',
                        input_shape=(img_channels, img_rows, img_cols), activation="relu", name="conv1"))
first_layer = model.layers[-1]
# this is a placeholder tensor that will contain our generated images
input_img = first_layer.input
model.add(Convolution2D(32, 3, 3, activation="relu", name="conv2"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, activation="relu", name="conv3"))
model.add(Convolution2D(64, 3, 3, activation="relu", name="conv4"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation="relu", name="fc1"))
model.add(Dropout(0.5))
model.add(Dense(nb_classes, activation="softmax", name="fc2"))

# let's train the model using SGD + momentum (how original).
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# get the symbolic outputs of each "key" layer (we gave them unique names).
layer_dict = dict([(layer.name, layer) for layer in model.layers])

model.fit(X_train, Y_train,batch_size=batch_size,nb_epoch=nb_epoch,validation_data=(X_test, Y_test),shuffle=True)

# util function to convert a tensor into a valid image
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

from keras import backend as K
from scipy.misc import imsave
import numpy as np
layer_names = ['conv1', 'conv2', 'conv3', 'conv4', 'fc1', 'fc2']
for layer_name in layer_names:
	for filter_index in range(8):
		# build a loss function that maximizes the activation
		# of the nth filter of the layer considered
		layer_output = layer_dict[layer_name].output
		loss = K.mean(layer_output[:, filter_index, :, :])

		# compute the gradient of the input picture wrt this loss
		grads = K.gradients(loss, input_img)[0]

		# normalization trick: we normalize the gradient
		grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

		# this function returns the loss and grads given the input picture
		iterate = K.function([input_img], [loss, grads])

		# we start from a gray image with some noise
		input_img_data = np.random.random((1, 3, img_width, img_height)) * 20 + 128.
		# run gradient ascent for 20 steps
		for i in range(20):
		    loss_value, grads_value = iterate([input_img_data])
		    input_img_data += grads_value * i

		img = input_img_data[0]
		img = deprocess_image(img)
		imsave('plots/%s_filter_%d.png' % (layer_name, filter_index), img)