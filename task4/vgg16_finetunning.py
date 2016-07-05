from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.datasets import cifar10, mnist
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from utils.datasets import terrassa


nb_classes=10
batch_size = 128
nb_epoch = 5
data_augmentation = False


# def preprocess_image_batch(image_paths, img_size=None, crop_size=None, color_mode="rgb", out=None):
#     img_list = []
    
#     for im_path in image_paths:
#         img = imread(im_path, mode='RGB')
#         if img_size:
#             img = imresize(img,img_size)
            
#         img = img.astype('float32')
#         # We permute the colors to get them in the BGR order
#         if color_mode=="bgr":
#             img[:,:,[0,1,2]] = img[:,:,[2,1,0]]
#         # We normalize the colors with the empirical means on the training set
#         img[:, :, 0] -= 123.68 
#         img[:, :, 1] -= 116.779
#         img[:, :, 2] -= 103.939
#         img = img.transpose((2, 0, 1))

#         if crop_size:
#             img = img[:,(img_size[0]-crop_size[0])//2:(img_size[0]+crop_size[0])//2
#                       ,(img_size[1]-crop_size[1])//2:(img_size[1]+crop_size[1])//2]
            
#         img_list.append(img)

#     img_batch = np.stack(img_list, axis=0)
#     if not out is None:
#         out.append(img_batch)
#     else:
#         return img_batch

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

(X_train, y_train), (X_test, y_test) = terrassa.load_data()

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

nb_classes = len(set(y_test))
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255


model=VGG_16("./weights/vgg16_weights.h5")
# model = convnet('VGG_16',weights_path="./weights/vgg16_weights.h5", heatmap=False)

model.layers.pop() 

for layer in model.layers:
    layer.trainable=False 

layer_last=Dense(1000, activation='softmax')
layer_last.trainable=True

model.add(layer_last)


sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='mse', metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath="./temporal_weights/weights_finetunned_alexnet_terrassa.hdf5", verbose=1, save_best_only=True)

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


