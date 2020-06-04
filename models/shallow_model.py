import os
import numpy as np

from keras.models import Sequential, Model
# from keras.layers import Merge, Input
from keras.layers import Input,DepthwiseConv2D,AveragePooling2D,SeparableConv2D,SpatialDropout2D,TimeDistributed,Bidirectional,ConvLSTM2D
from keras.layers.core import Dense, Dropout, Activation, Flatten, Lambda, Reshape, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Conv3D, MaxPooling3D,Conv1D
from keras.layers.recurrent import GRU, LSTM
from keras.layers.advanced_activations import ELU
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.optimizers import SGD, Adam, Adagrad, Adadelta, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.constraints import max_norm
from keras import backend as K
from keras.preprocessing import image
import math
from models.customCallbacks import MyEarlyStopping, MyModelCheckpoint
from keras.layers import Reshape
from keras import backend as K
K.set_image_data_format('channels_last')



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
    #nhan model for 7s
    def setup_3(self, X_train_shape):
        # print ('X_train shape', X_train_shape)
        # Input shape = (None,13,2,126,1)
        inputs = Input(shape=X_train_shape[1:])

        normal1 = BatchNormalization(
            axis=2,
            name='normal1')(inputs)

        convlstm1 = ConvLSTM2D(
            filters=16,
            kernel_size=(X_train_shape[2], 3),
            padding='valid', strides=(1, 2),
            activation='tanh',
            dropout=0.0, recurrent_dropout=0.0,
            return_sequences=True,
            name='convlstm1')(normal1)

        convlstm2 = ConvLSTM2D(
            filters=32,
            kernel_size=(1, 3),
            padding='valid', strides=(1, 2),
            activation='tanh',
            dropout=0.0, recurrent_dropout=0.0,
            return_sequences=True,
            name='convlstm2')(convlstm1)

        convlstm3 = ConvLSTM2D(
            filters=64,
            kernel_size=(1, 3),
            padding='valid', strides=(1, 2),
            activation='tanh',
            dropout=0.0, recurrent_dropout=0.0,
            return_sequences=False,
            name='convlstm3')(convlstm2)

        flat = Flatten()(convlstm3)

        drop1 = Dropout(0.5)(flat)

        dens1 = Dense(256, activation='sigmoid', name='dens1')(drop1)
        drop2 = Dropout(0.5)(dens1)

        dens2 = Dense(self.nb_classes, name='dens2')(drop2)

        # option to include temperature in softmax
        temp = 1.0
        temperature = Lambda(lambda x: x / temp)(dens2)
        last = Activation('softmax')(temperature)

        self.model = Model(input=inputs, output=last)

        adam = Adam(lr=5e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        self.model.compile(loss='binary_crossentropy',
                           optimizer=adam,
                           metrics=['accuracy'])

        #print(self.model.summary())
        return self

    # Define model 2
    def setup1(self, X_train_shape):

        dropoutType = 'Dropout'
        dropoutRate = 0.5
        norm_rate = 0.25
        nb_classes = 2

        if dropoutType == 'SpatialDropout2D':
            dropoutType = SpatialDropout2D
        elif dropoutType == 'Dropout':
            dropoutType = Dropout
        else:
            raise ValueError('dropoutType must be one of SpatialDropout2D '
                             'or Dropout, passed as a string.')

        input1 = Input(batch_shape=(None, 30, 26, 1, 20))

        normal1 = BatchNormalization(
            axis=-1,
            name='normal1')(input1)
        ##################################################################
        block1 = ConvLSTM2D(filters=16, kernel_size=(3, 1),
                            padding='valid', strides=(2, 1), activation='tanh', dropout=0.5, return_sequences=True)(
                             normal1)
        block2 = ConvLSTM2D(filters=32, kernel_size=(3, 1),
                            padding='valid', strides=(2, 1), activation='tanh', dropout=0.5, return_sequences=True)(
                              block1)
        block3 = ConvLSTM2D(filters=16, kernel_size=(3, 1),
                            padding='valid', strides=(2, 1), activation='tanh', dropout=0.5, return_sequences=False)(
                             block2)
        # block2 = Reshape((30, 25))(block2)
        # block2 = GRU(25, return_sequences=True, activation='elu')(block2)
        # block2 = TimeDistributed(Dense(2, activation='sigmoid'))(block2)
        # last = block2
        # print(block2)
        # block2 = Reshape((10, 64,1))(block2)
        flatten = Flatten(name='flatten')(block3)
        dense1 = Dense(128, name='dense1',
                       )(flatten)
        dense = Dense(nb_classes, name='dense2',
                      )(dense1)

        last = Activation('sigmoid')(dense)

        # softmax = Activation('softmax', name='softmax')(dense)

        self.model = Model(input=input1, output=last)

        adam = Adam(lr=5e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        self.model.compile(loss='binary_crossentropy',
                      optimizer=adam,
                      metrics=['accuracy'])

        print(self.model.summary())
        return self





    def setup(self, X_train_shape):

        dropoutType = 'Dropout'
        dropoutRate = 0.5
        norm_rate = 0.25
        nb_classes = 2

        if dropoutType == 'SpatialDropout2D':
            dropoutType = SpatialDropout2D
        elif dropoutType == 'Dropout':
            dropoutType = Dropout
        else:
            raise ValueError('dropoutType must be one of SpatialDropout2D '
                             'or Dropout, passed as a string.')

        input1 = Input(batch_shape=(None, 30, 20, 26, 1))

        normal1 = TimeDistributed(BatchNormalization(
            axis=1,
            name='normal1'), input_shape=(30, 20, 26, 1))(input1)
        ##################################################################
        block1 = TimeDistributed(Convolution2D(16, (3, 3),
                                               padding='same', strides=(1, 1), activation='elu', name='conv1'),
                                 input_shape=(30, 20, 26, 1))(normal1)

        con1 = TimeDistributed(MaxPooling2D((2, 2)), input_shape=(30, 20, 26, 16))(block1)
        normal2 = TimeDistributed(BatchNormalization(axis=1), input_shape=(30, 10, 13, 16))(con1)
        block2 = TimeDistributed(Convolution2D(32, (3, 3),
                                               padding='same', strides=(1, 1), activation='elu', name='conv2'),
                                 input_shape=(30, 10, 13, 16))(normal2)

        con2 = TimeDistributed(MaxPooling2D((4, 4)), input_shape=(30, 10, 13, 32))(block2)
        normal2 = TimeDistributed(BatchNormalization(axis=1), input_shape=(30, 2, 3, 32))(con2)
        flatten = TimeDistributed(Flatten(name='flatten'), input_shape=(30, 2, 3, 32))(normal2)
        block3 = Conv1D(32, 10)(flatten)
        block4 = Bidirectional(LSTM(24, return_sequences=True, activation='elu'))(block3)
        block6 = Bidirectional(LSTM(12, return_sequences=False, activation='elu'))(block4)
        block7 = dropoutType(dropoutRate)(block6)
        # block2 = Reshape((30, 25))(block2)
        # block2 = GRU(25, return_sequences=True, activation='elu')(block2)
        # block2 = TimeDistributed(Dense(2, activation='sigmoid'))(block2)
        # last = block2
        # print(block2)
        # block2 = Reshape((10, 64,1))(block2)
        #flatten = Flatten(name='flatten')(block7)

        dense = Dense(nb_classes, name='dense',
                      )(block7)

        last = Activation('sigmoid')(dense)

        # softmax = Activation('softmax', name='softmax')(dense)

        self.model = Model(input=input1, output=last)

        adam = Adam(lr=5e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        self.model.compile(loss='binary_crossentropy',
                      optimizer=adam,
                      metrics=['accuracy'])

        #print(self.model.summary())
        return self






    def fit(self, training_generator, validation_generator,class_weights_train):
        early_stop = MyEarlyStopping(patience=20, verbose=0)
        checkpointer = MyModelCheckpoint(
            filepath="2channel_sftf_7_train_whole.h5",
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
    def fit_3(self, training_generator, validation_generator,class_weights_train):
        early_stop = MyEarlyStopping(patience=20, verbose=0)
        #2channel_sftf_3_train_whole
        checkpointer = MyModelCheckpoint(
            filepath="2channel_sftf_3_train.h5",
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
    def fit_5(self, training_generator, validation_generator,class_weights_train):
        early_stop = MyEarlyStopping(patience=20, verbose=0)
        #2channel_sftf_5_train_whole.h5
        checkpointer = MyModelCheckpoint(
            filepath="2channel_sftf_5_train.h5",
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


    def fit_7(self, training_generator, validation_generator,class_weights_train):
        early_stop = MyEarlyStopping(patience=20, verbose=0)
        checkpointer = MyModelCheckpoint(
            filepath="2channel_sftf_7_train.h5",
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


    def load_trained_weights(self, space_time):
        #2channel_sftf_5_train_whole.h5
        self.model.load_weights("2channel" + str(space_time) + "_train2" + ".h5")
        #self.model.load_weights(filename)
        print('Loading pre-trained weights from %s.' % "2channel" + str(space_time) + "_train" + ".h5")
        return self

    def predict_proba(self, X):
        return self.model.predict([X])

    def evaluate_1(self,X,total_time,space_time):
        predictions = self.model.predict(X, verbose=0)[:, 1]
        print(total_time)
        predictions1 = []
        if total_time<2*space_time-1:
            for i in range(total_time):
                if i+1 <=space_time-1:
                    predictions1.append(np.mean(predictions[:i + 1]))
                if space_time-2<i<2*space_time-1:
                    predictions1.append(np.mean(predictions[i-total_time:]))
        else:
            for i in range(total_time):
                if i+1 <=space_time-1:
                    predictions1.append(np.mean(predictions[:i + 1]))
                if space_time-2<i<total_time-(space_time-1):
                    predictions1.append(np.mean(predictions[i-(space_time-1):i+1]))
                if i >=total_time-(space_time-1):
                    predictions1.append(np.mean(predictions[i-total_time:]))

        predictions1 = np.array(predictions1, dtype=np.float)
        return predictions1

    def vote_wirte(self,result_list,file_name):
        print(len(result_list[0]),len(result_list[1]),len(result_list[2]))
        print('begin vote predciction')
        predictions3 = result_list[0]
        predictions5 = result_list[1]
        predictions7 = result_list[2]
        predictions = []
        for m in range(len(predictions3)):
            predictions.append(np.bincount([predictions3[m],predictions5[m],predictions7[m]]).argmax())
        begin = predictions[0]
        index_begin = 0
        print('begin write '+file_name)
        for i in range(len(predictions)):
            if i == len(predictions)-1:
                index_end=(i+1)
                print('---------')
                print(index_begin,index_end)
                self.write_file(file_name, index_begin,index_end, begin)
            if predictions[i]!=begin:
                index_end = i
                print('---------')
                print(index_begin, index_end)
                self.write_file(file_name, index_begin, index_end, begin)
                begin = predictions[i]
                index_begin = i
        print('end write ' + file_name)




    def write_file(self,file_name,index_begin,index_end,begin):
        f = open("result_eval_vote_2ch_f3f7_0.95.txt", 'a')

        if begin == 1:
            seiz_name = 'seiz'
            confidence = 0.90
            print(file_name + ' ' + str(round(index_begin ,4)) + ' ' + str(round(index_end,4)) + ' '+ str(
                round(confidence, 4))
                 + '\n')
            f.write(file_name + ' ' + str(round(index_begin,4)) + ' ' + str(round(index_end,4)) + ' '+ str(
                round(confidence, 4))
                 + '\n')

        f.close()