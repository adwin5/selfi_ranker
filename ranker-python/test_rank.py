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
from keras.models import load_model

import cv2, numpy as np
import matplotlib.pyplot as plt
from keras.applications.inception_v3 import InceptionV3
import selfie_data_generator as data_g


def read_test_image(img_name, folder="/home/adwin/Desktop/selfi-cv/Selfie-dataset/images/", expand=True, process=True):
    image_URL = folder+img_name+".jpg"
    im = cv2.resize(cv2.imread(image_URL), (224, 224)).astype(np.float32)
    if process == True:
        im[:,:,0] -= 103.939
        im[:,:,1] -= 116.779
        im[:,:,2] -= 123.68
    im = im.transpose((2,0,1))
    if expand == True:
        im = np.expand_dims(im, axis=0)
    return im
# predict result based on image object
def test(im):
    # build model
    model = load_model("weights/selfi-01-0.28.f5")

    # compile model
    sgd = SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
    #adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=sgd, loss='mean_squared_error')
    score = model.predict(im)
    return score
def color_RGB(img_name, folder="/home/adwin/Desktop/selfi-cv/Selfie-dataset/images/"):
    image_URL = folder+img_name+".jpg"
    im = cv2.resize(cv2.imread(image_URL), (224, 224))
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
# randomly pick samples to validate
def random_validate(n=10):
    vals_copy = np.copy(data_g.vals) 
    np.random.shuffle(vals_copy)
    random_valid_name, random_valid_GT = vals_copy[:n][:,0], vals_copy[:n][:,1]
    print random_valid_name
    random_valid_inputs = np.zeros((n,3,224,224))
    for i in range(n):
        random_valid_inputs[i] = read_test_image(random_valid_name[i],\
                        folder="/home/adwin/Desktop/selfi-cv/Selfie-dataset/images/",expand=False, process=False)
    
    model = load_model("weights/selfi-01-0.28.f5")
    # compile model
    sgd = SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
    #adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=sgd, loss='mean_squared_error')
    score = model.predict(random_valid_inputs)
    print score
    fig = plt.figure(figsize=(10, 6))

    ax1 = plt.subplot(2,5,1)
    ax2 = plt.subplot(2,5,2)
    ax3 = plt.subplot(2,5,3)
    ax4 = plt.subplot(2,5,4)
    ax5 = plt.subplot(2,5,5)
    ax6 = plt.subplot(2,5,6)
    ax7 = plt.subplot(2,5,7)
    ax8 = plt.subplot(2,5,8)
    ax9 = plt.subplot(2,5,9)
    ax10 = plt.subplot(2,5,10)

    ax1.imshow(color_RGB(img_name =random_valid_name[0]))
    ax1.set_title("Selfi-score : \n"+ str(score[0]))
    ax2.imshow(color_RGB(img_name =random_valid_name[1]))
    ax2.set_title("Selfi-score : \n"+ str(score[1]))
    ax3.imshow(color_RGB(img_name =random_valid_name[2]))
    ax3.set_title("Selfi-score : \n"+ str(score[2]))
    ax4.imshow(color_RGB(img_name =random_valid_name[3]))
    ax4.set_title("Selfi-score : \n"+ str(score[3]))
    ax5.imshow(color_RGB(img_name =random_valid_name[4]))    
    ax5.set_title("Selfi-score : \n"+ str(score[4]))

    ax6.imshow(color_RGB(img_name =random_valid_name[5]))
    ax6.set_title("Selfi-score : \n"+ str(score[5]))
    ax7.imshow(color_RGB(img_name =random_valid_name[6]))
    ax7.set_title("Selfi-score : \n"+ str(score[6]))
    ax8.imshow(color_RGB(img_name =random_valid_name[7]))
    ax8.set_title("Selfi-score : \n"+ str(score[7]))
    ax9.imshow(color_RGB(img_name =random_valid_name[8]))
    ax9.set_title("Selfi-score : \n"+ str(score[8]))
    ax10.imshow(color_RGB(img_name =random_valid_name[9]))    
    ax10.set_title("Selfi-score : \n"+ str(score[9]))

    plt.show()

if __name__ == "__main__":
    im = read_test_image(img_name="10004275_818718201490015_1239261657_a",\
            folder="/home/adwin/Desktop/selfi-cv/Selfie-dataset/images/",expand=True, process=True)
    #print test(im)
    random_validate()
    # Test pretrained model
    #model = VGG_16('vgg16_weights.h5')

