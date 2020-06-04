# Adapt from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
import numpy as np
import keras
import os
from sklearn import preprocessing

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=2, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        # 'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size), dtype=int)


        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            x_ = np.load(ID) # shape (1, 19, 750)

            # print ('__data_generation', x_.shape)
            #x_ = np.expand_dims(x_, axis=0)
            #x_ = preprocessing.normalize(x_)
            # 20,30,26
            #x_ = np.moveaxis(x_, 1, 0)
            #30,20,26
            #for k in range(30):
                #x_[k] = preprocessing.normalize(x_[k],axis=0)
            #x_ = np.moveaxis(x_, 1, -1)
            #30,26,20
            #13,2,126,1
            x_=np.moveaxis(x_, 0, -1)
            #F3-F7, P3-O1
            #['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'T3', 'C3', 'CZ', 'C4', 'T4', 'T5', 'P3', 'PZ', 'P4', 'T6',
             #'O1', 'O2']
            #x_ = x_.astype(float)
            #diff1= x_[:,3:4,:,:]-x_[:,2:3,:,:]
            #print(i,'done')
            #diff2 = x_[:,13:14,:,:]-x_[:,17:18,:,:]
            #x_ = np.concatenate((diff1,diff2), axis= 1)

            #13,2,126,1
            X[i,] = x_
            #30,26,1,20
            #print('-----------------')
            #print(x_.shape)
            #X[i,] =x_
            #X[i,] = np.transpose(x_,(0,1,3,2)) # shape (1,5,126,20)
            # print ('__data_generation transpose', X[i,].shape)

            # Store class
            # Store class
            label = ID.strip().split('_')
            # label = label[-1].strip().split('.')[0]
            label = label[-2]

            if label=='seiz':
                y[i]= 1
            elif label == 'bckg':
                y[i]= 0

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
        #return X, y
        # X = []
        # y = []
        # for i, ID in enumerate(list_IDs_temp):
        #     # Store sample
        #     x_ = np.load(ID) # shape (1, 5, 20, 126)
        #     x_ = np.expand_dims(x_, axis=0) # (1, 1, 5, 20, 126)
        #     print ('__data_generation', x_.shape)
        #     X.append(np.transpose(x_,(0,1,2,4,3))) # shape (1,1,5,126,20)
        #     print ('__data_generation transpose', X[-1].shape)

        #     # Store class
        #     label = ID.strip().split('_')
        #     label = label[-1].strip().split('.')[0]
        #     if label=='seiz':
        #         y.append(1)
        #     elif label == 'bckg':
        #         y.append(0)
        # X = np.concatenate(X, axis=0)
        # y = np.array(y)
        # assert X.shape[0] == self.batch_size
        # return X, keras.utils.to_categorical(y, num_classes=self.n_classes)