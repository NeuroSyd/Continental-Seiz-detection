import os
import numpy as np

from keras import activations
from keras.models import Sequential, Model
#from keras.layers import Merge, Input
from keras.layers import  Input, ConvLSTM2D
from keras.layers.core import Dense, Dropout, Activation, Flatten, Lambda, Reshape, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Conv3D, MaxPooling3D
from keras.layers.recurrent import GRU, LSTM
from keras.layers.advanced_activations import ELU
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.optimizers import SGD,Adam,Adagrad,Adadelta,RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.constraints import max_norm
from keras import backend as K
from keras.preprocessing import image
import math
from models.customCallbacks import MyEarlyStopping, MyModelCheckpoint
from keras.layers import Reshape
from keras import backend as K



class ConvLstmNet(object):
    
	def __init__(self,batch_size=16,nb_classes=2,epochs=2):

		self.batch_size = batch_size
		self.nb_classes = nb_classes
		self.epochs = epochs

    # Define custom loss
    
	def setup(self,X_train_shape):
		#print ('X_train shape', X_train_shape)
		# Input shape = (None,5,20,126,1)
		inputs = Input(shape=X_train_shape[1:])

		normal1 = BatchNormalization(
			axis=2,
			name='normal1')(inputs)
		
		convlstm1 = ConvLSTM2D(
			filters=16,
			kernel_size=(X_train_shape[2], 3),
			padding='valid',strides=(1,2),
            activation='tanh',
            dropout=0.0, recurrent_dropout=0.0,
            return_sequences=True,
			name='convlstm1')(normal1)

		convlstm2 = ConvLSTM2D(
			filters=32,
			kernel_size=(1, 3),
			padding='valid',strides=(1,2),
            activation='tanh',
            dropout=0.0, recurrent_dropout=0.0,
            return_sequences=True,
			name='convlstm2')(convlstm1)

		convlstm3 = ConvLSTM2D(
			filters=64,
			kernel_size=(1, 3),
			padding='valid',strides=(1,2),
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

		print (self.model.summary())
		return self


	def fit(self,training_generator, validation_generator, class_weights, weightpath="weights_best.h5"):		
		
		early_stop = MyEarlyStopping(patience=10, verbose=0)
		checkpointer = MyModelCheckpoint(
			filepath=weightpath,
			verbose=0, save_best_only=True)
		

		self.model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
					epochs=self.epochs,
                    class_weight=class_weights,
                    use_multiprocessing=False, #workers=2,
					callbacks=[early_stop, checkpointer])

		# self.model.load_weights(weightpath)
		#if self.mode == 'cv':
			#os.remove("weights_%s_%s.h5" %(self.target, self.mode))
		return self

	def load_trained_weights(self, filename):
		self.model.load_weights(filename)
		print ('Loading pre-trained weights from %s.' %filename)
		return self

	def predict_proba(self,X):
		return self.model.predict(X)

	def evaluate(self, X, y):
		predictions = self.model.predict(X, verbose=0)[:,1]
		from sklearn.metrics import roc_auc_score
		auc_test = roc_auc_score(y, predictions)
		print('Test AUC is:', auc_test)


class ConvLstmNetDeep(ConvLstmNet):
    def __init__(self,batch_size=16,nb_classes=2,epochs=2):
        super().__init__(batch_size,nb_classes,epochs)
    
    def setup(self,X_train_shape):
        #print ('X_train shape', X_train_shape)
        # Input shape = (None,5,19,126,1)
        inputs = Input(shape=X_train_shape[1:])

        normal1 = BatchNormalization(
            axis=2,
            name='normal1')(inputs)

        convlstm1 = ConvLSTM2D(
            filters=16,
            kernel_size=(X_train_shape[2], 5),
            padding='valid',strides=(1,1),
            activation='tanh',
            dropout=0.2, recurrent_dropout=0.2,
            return_sequences=True,
            name='convlstm1')(normal1)

        convlstm2 = ConvLSTM2D(
            filters=32,
            kernel_size=(1, 3),
            padding='valid',strides=(1,2),
            activation='tanh',
            dropout=0.2, recurrent_dropout=0.2,
            return_sequences=True,
            name='convlstm2')(convlstm1)

        convlstm3 = ConvLSTM2D(
            filters=64,
            kernel_size=(1, 3),
            padding='valid',strides=(1,2),
            activation='tanh',
            dropout=0.2, recurrent_dropout=0.2,
            return_sequences=True,
            name='convlstm3')(convlstm2)	

        convlstm4 = ConvLSTM2D(
            filters=128,
            kernel_size=(1, 3),
            padding='valid',strides=(1,2),
            activation='tanh',
            dropout=0.2, recurrent_dropout=0.2,
            return_sequences=False,
            name='convlstm4')(convlstm3)	


        flat = Flatten()(convlstm4)

        drop1 = Dropout(0.5)(flat)

        dens1 = Dense(1024, activation=activations.elu, name='dens1')(drop1)
        drop2 = Dropout(0.5)(dens1)
        dens2 = Dense(256, activation=activations.elu, name='dens2')(drop2)
        drop3 = Dropout(0.5)(dens2)

        dens3 = Dense(self.nb_classes, name='dens3')(drop3)

        # option to include temperature in softmax
        # temp = 1.0
        # temperature = Lambda(lambda x: x / temp)(dens3)
        last = Activation('softmax')(dens3)

        self.model = Model(input=inputs, output=last)

        adam = Adam(lr=5e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        self.model.compile(loss='binary_crossentropy',
                        optimizer=adam,
                        metrics=['accuracy'])

        print (self.model.summary())
        return self
    def fit(self, training_generator, validation_generator,class_weights_train):
        early_stop = MyEarlyStopping(patience=10, verbose=0)
        checkpointer = MyModelCheckpoint(
            filepath="2channel_sftf_3_deepconv_train2.h5",
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
        #self.model.load_weights("2channel_" +str(space_time)+'deep' + "_train2" + ".h5")
        self.model.load_weights("2channel_"+str(space_time)+"_deep_train2.h5")

        #self.model.load_weights(filename)
        #2channel_sftf_3_deepconv_train2.h5
        print('Loading pre-trained weights from %s.' % "2channel" + str(space_time) + "_train2" + ".h5")
        return self
    def fit1(self, training_generator, validation_generator,class_weights_train):
        early_stop = MyEarlyStopping(patience=10, verbose=0)
        checkpointer = MyModelCheckpoint(
            filepath="2channel_sftf_5_deepconv_train2.h5",
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
    def fit2(self, training_generator, validation_generator,class_weights_train):
        early_stop = MyEarlyStopping(patience=10, verbose=0)
        checkpointer = MyModelCheckpoint(
            filepath="2channel_sftf_7_deepconv_train2.h5",
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
    def evaluate_1(self,X,total_time,space_time):
        #self.model.load_weights("2channel"+str(space_time)+"_train2"+".h5")
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

        print(len(predictions1))
        predictions1 = np.array(predictions1, dtype=np.float)
        return predictions1