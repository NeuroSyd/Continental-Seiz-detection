import os
import numpy as np
from keras import activations
from keras.models import Sequential, Model
# from keras.layers import Merge, Input
from keras.layers import Input,DepthwiseConv2D,AveragePooling2D,SeparableConv2D,SpatialDropout2D,TimeDistributed,Bidirectional
from keras.layers.core import Dense, Dropout, Activation, Flatten, Lambda, Reshape, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Conv3D, MaxPooling3D
from keras.layers.recurrent import GRU, LSTM
from keras.layers.advanced_activations import ELU
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.optimizers import SGD, Adam, Adagrad, Adadelta, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.constraints import max_norm
from vis.visualization import overlay,visualize_cam
#from models.visualize_cam import  visualize_cam
from vis.utils import utils
import matplotlib.image as mpimg
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from keras import backend as K
from keras.preprocessing import image
import math
from models.customCallbacks import MyEarlyStopping, MyModelCheckpoint
from keras.layers import Reshape
from keras import backend as K
import mne
K.set_image_data_format('channels_first')



# Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
def loss(y_true, y_pred):
    # y_pred = [max(min(pred[0], 1-K.epsilon()), K.epsilon()) for pred in y_pred]
    y_pred = K.maximum(K.minimum(y_pred, 1 - K.epsilon()), K.epsilon())
    t_loss = (-1) * (K.exp(y_true) * y_true * K.log(y_pred) + (1 - y_true) * K.log(1 - y_pred) / K.exp(y_pred))

    return K.mean(t_loss)


class ConvNN(object):

    def __init__(self, batch_size=16, nb_classes=2, epochs=2):
        self.batch_size = batch_size
        self.nb_classes = nb_classes
        self.epochs = epochs

    # Define custom loss

    def setup(self, X_train_shape):

        nb_classes = 2
        Chans = 19
        Samples = 250
        dropoutRate = 0.5
        kernLength = 16
        F1 = 2,
        D = 2
        F2 = 16,
        norm_rate = 0.25,
        dropoutType = 'Dropout'



        if dropoutType == 'SpatialDropout2D':
            dropoutType = SpatialDropout2D
        elif dropoutType == 'Dropout':
            dropoutType = Dropout
        else:
            raise ValueError('dropoutType must be one of SpatialDropout2D '
                             'or Dropout, passed as a string.')

        input1 = Input(batch_shape=(None,1,19,750))

        normal1 = BatchNormalization(
            axis=1,
            name='normal1')(input1)
        ##################################################################
        block1 = Convolution2D(4, (1, 125),
                              padding='same', use_bias=False, name='conv1')(normal1)
        #block1 = Convolution2D(F1, (1, 4), padding='same',strides=(1, 1))(normal1)

        block1 = BatchNormalization(axis=1)(block1)
        block1 = DepthwiseConv2D((19, 1), use_bias=False,
                                 depth_multiplier=D,
                                 depthwise_constraint=max_norm(1.))(block1)
        # print(block1)
        block1 = BatchNormalization(axis=1)(block1)
        block1 = Activation('elu')(block1)
        block1 = AveragePooling2D((1, 6))(block1)
        block1 = dropoutType(dropoutRate)(block1)

        block1 = SeparableConv2D(60, (1, 16),
                               padding='same', use_bias=False ,name='conv2')(block1)
        block2 = BatchNormalization(axis=1)(block1)
        block2 = Activation('elu')(block2)
        block2 = AveragePooling2D((1, 10))(block2)
        block2 = dropoutType(dropoutRate)(block2)
        #print(block2)
        #block2 = Reshape((30,24))(block2)
        #block2 = Bidirectional(GRU(24, return_sequences=False,activation='elu'))(block2)
        #block2 = TimeDistributed(Dense(2, activation='sigmoid'))(block2)
        #last = block2
        flatten = Flatten(name='flatten')(block2)

        dense = Dense(nb_classes, name='visualized_layer',
                      kernel_constraint=max_norm(norm_rate))(flatten)

        last = Activation('sigmoid')(dense)
        softmax = Activation('softmax', name='softmax')(dense)


        self.model = Model(input=input1, output=last)

        adam = Adam(lr=5e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        self.model.compile(loss='binary_crossentropy',
                           optimizer=adam,
                           metrics=['accuracy'])

        print(self.model.summary())
        return self


    def fit(self, training_generator, validation_generator,class_weights_train):
        early_stop = MyEarlyStopping(patience=10, verbose=0)
        checkpointer = MyModelCheckpoint(
            filepath="weights_best.h5",
            verbose=0, save_best_only=True)
        #class_weight = {0: 20.,
                        #1: 1,
                        #}
        self.model.fit_generator(generator=training_generator,
                                 validation_data=validation_generator,
                                 epochs=self.epochs,class_weight=class_weights_train,
                                 use_multiprocessing=False, workers=4,
                                 callbacks=[early_stop,checkpointer])

        #self.model.load_weights("weights_best.h5")
        # if self.mode == 'cv':
        # os.remove("weights_%s_%s.h5" %(self.target, self.mode))
        return self

    def load_trained_weights(self, filename):
        self.model.load_weights(filename)
        print('Loading pre-trained weights from %s.' % filename)
        return self

    def predict_proba(self, X):
        return self.model.predict([X])

    def evaluate(self, X,y,file_name):
        self.model.load_weights("weights_best_2.h5")
        predictions = self.model.predict(X, verbose=0)[:, 1]
        #print(predictions)
        # Find the index of the to be visualized layer above
        layer_index = utils.find_layer_idx(self.model, 'visualized_layer')
        #make the color blue
        #ch_types = ['mag']*18
        ch_names = ['0', '1', '2', '3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18']
        #ch_names = ['0','1','2','3']
        sfreq = 250
        ch_types = ['eeg']*19
        info = mne.create_info(ch_names=ch_names,ch_types=ch_types,sfreq=sfreq)
        # Swap softmax with linear

        for i in range(len(y)):
            max = 0
            max_index = -1
            max_list = []
            input_image = X[i]
            input_class = y[i]
            file_na = file_name[i]
            print(file_na)
            self.model.layers[layer_index].activation = activations.linear
            self.model = utils.apply_modifications(self.model)
            visualization = visualize_cam(self.model, layer_index, filter_indices=input_class,
                                          seed_input=input_image)
            #print(input_image.shape)
            #print(visualization.shape

            for i in range(19):
                print(np.array_equal(visualization[0], visualization[i]))
                #print(visualization[i][:100])
                if np.amax(visualization[i]) >max:
                    max = np.amax(visualization[i])
                    #print(max)
                    max_index=i
                max_list.append(np.amax(visualization[i]))
            #print('max_index',i)
            #print(max_list)
            raw = mne.io.RawArray(input_image[0,:,:], info)
            raw.plot(n_channels=19,  title='Data from arrays',color='b',scalings=0.08,
                     show=True, block=True)
            plt.savefig('./grad_cam/' + str(file_na) + str(input_class) + '.png')
            raw1 = mne.io.RawArray(visualization, info)
            raw1.plot(n_channels=19, title='important area value', color='r', scalings=0.5,
                     show=True, block=True)
            plt.savefig('./grad_cam/' + str(file_na) + str(input_class)+str(predictions[i])+'area importance' + '.png')
            #fig, axes = plt.subplots(2, 1)
            #axes[0].imshow(input_image[0,0,:])
            #axes[0].set_title('Input')
            #axes[1].imshow(visualization[0,:])
            #axes[1].set_title('Grad-CAM')
            #heatmap = np.uint8(cm.jet(visualization[0,:]))
            #original = np.uint8(cm.gray(input_image[0,0,:]))
            #pic_overlay = overlay(heatmap, original)
            #axes[2].imshow(pic_overlay)
            #axes[2].set_title('Overlay')

            #mpimg.imsave('./grad_cam/' +str(file_na)+str(predictions[i])+ '.png',
                         #pic_overlay)

    def write_file(self,file_name,index_begin,index_end,begin):
        f = open("result_eval.txt", 'a')
        if begin == 0:
            seiz_name = 'bckg'
        else:
            seiz_name = 'seiz'
        print(file_name + ' ' + str(index_begin) + ' ' + str(index_end) + ' ' + seiz_name + ' ' + str(
            1.0000) + '\n')
        f.write(file_name + ' ' + str(index_begin) + ' ' + str(index_end) + ' ' + seiz_name + ' ' + str(
            1.0000) + '\n')
        f.close()