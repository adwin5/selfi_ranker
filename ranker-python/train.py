from keras import backend as K
K.set_image_dim_ordering('th')

from keras.models import Sequential, Model
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers import Input
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras import optimizers
from keras.callbacks import ModelCheckpoint

import cv2, numpy as np
import matplotlib.pyplot as plt
from keras.applications.inception_v3 import InceptionV3
import selfie_data_generator as data_g

def custom_inception_v3():
    model_v3 =  InceptionV3(include_top=False, weights='imagenet')
    x = model_v3.output
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(1024, activation='tanh')(x)
    score = Dense(1, name='score')(x)
    whole_model = Model(input=model_v3.input,output=score)
    for layer in model_v3.layers:
        layer.trainable = False
    # # we chose to train the top 2 inception blocks, i.e. we will freeze
    # # the first 172 layers and unfreeze the rest:
    # for i, layer in enumerate(whole_model.layers):
    #     print(i, layer.name)
    for layer in whole_model.layers[:172]:
        layer.trainable = False
    for layer in whole_model.layers[172:]:
        layer.trainable = True
    return whole_model    

def train(batch_size=32):
    # build model
    model =  custom_inception_v3()
    print model.summary()

    # compile model
    sgd = SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
    #adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=sgd, loss='mean_squared_error')
    # start traning
    nb_train_samples = len(data_g.trains)
    nb_validation_samples = len(data_g.vals)
    # checkpoint
    filepath="weights/selfi-{epoch:02d}-{val_loss:.2f}.f5"
    checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=True)
    callbacks_list = [checkpoint]

    print nb_train_samples, nb_validation_samples
    history = model.fit_generator(data_g.train_generator(),
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=10,
        validation_data=data_g.validation_generator(),
        validation_steps=nb_validation_samples // batch_size,
        callbacks=callbacks_list)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

if __name__ == "__main__":
    train()
    # Test pretrained model
    #model = VGG_16('vgg16_weights.h5')

