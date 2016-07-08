from keras.optimizers import SGD
from convnetskeras.convnets import preprocess_image_batch, convnet
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.datasets import cifar10, mnist
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from scipy.misc import imread, imresize
import numpy as np

nb_classes=10
batch_size = 128
nb_epoch = 5
data_augmentation = False


def preprocess_image_batch(image_paths, img_size=None, crop_size=None, color_mode="rgb", out=None):
    img_list = []
    
    for im_path in image_paths:
        img = imread(im_path, mode='RGB')
        if img_size:
            img = imresize(img,img_size)
            
        img = img.astype('float32')
        # We permute the colors to get them in the BGR order
        if color_mode=="bgr":
            img[:,:,[0,1,2]] = img[:,:,[2,1,0]]
        # We normalize the colors with the empirical means on the training set
        img[:, :, 0] -= 123.68 
        img[:, :, 1] -= 116.779
        img[:, :, 2] -= 103.939
        img = img.transpose((2, 0, 1))

        if crop_size:
            img = img[:,(img_size[0]-crop_size[0])//2:(img_size[0]+crop_size[0])//2
                      ,(img_size[1]-crop_size[1])//2:(img_size[1]+crop_size[1])//2]
            
        img_list.append(img)

    img_batch = np.stack(img_list, axis=0)
    if not out is None:
        out.append(img_batch)
    else:
        return img_batch

def VGG_16(weights_path=None):

    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model


# model.load_weights("./weights/alexnet_weights.h5")

# im = preprocess_image_batch(['examples/dog.jpg'],img_size=(256,256), crop_size=(227,227), color_mode="rgb")

# model = convnet('alexnet',weights_path="./weights/alexnet_weights.h5", heatmap=False, trainable=False)

model=VGG_16("/home/deep_learning_seminar/dlcv04/task4/weights/vgg16_weights.h5")
# model = convnet('VGG_16',weights_path="./weights/vgg16_weights.h5", heatmap=False)

# model.layers.pop() 

# for layer in model.layers:
#     layer.trainable=False 

# layer_last=Dense(1000, activation='softmax')
# layer_last.trainable=True

# model.add(layer_last)


sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='mse', metrics=['accuracy'])
out = model.predict_proba(preprocess_image_batch(['/home/deep_learning_seminar/dlcv04/utils/alexnet/examples/kitten_crop_face.jpg'],img_size=(224,224)))
print np.argmax(out)
print out