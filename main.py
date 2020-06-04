import json
import os
import glob
import os.path
import numpy as np
import keras
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
keras.backend.set_image_data_format('channels_last')
print ('Using Keras image_data_format=%s' % keras.backend.image_data_format())
from utils.prep_data import train_val_split_file_list,get_path,get_ref_evalu_df,get_image_path,get_ref_evalu_lcc,get_ref_evalu_stft,predict,\
    vote_wirte,write_file,write,train_val_get_file_list,train_val_split_diff_patient_file_list
# from utils.load_data import load_npy
from models.shallow_model import ConvNN
#from models.CNN_2ch_3s import ConvNN
#from models.conv_lstm import ConvLstmNetDeep
# import matplotlib.pyplot as plt


from keras.models import Sequential
from utils.data_gen import DataGenerator

#train_path = './test_data/'
train_path = '/mnt/data7_M2/tempdata/tuh_5s_stft_2ch_O1P3_F3F7_train_whole'
#dev_path = '/mnt/data7_M2/tempdata/tuh_5s_stft_2ch_O1P3_F3F7_dev_whole'
evaluate_path_all = '/mnt/data4/datasets/seizure/TUH_EEG_Seizure/v1.5.1/edf/eval/'
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--mode", help="train or test. cv is for leave-one-out cross-validation")
args = parser.parse_args()

if __name__ == "__main__":



    if args.mode =='train':
        #print('begin training mode')

        #fl_train, fl_val,class_weights_train= train_val_get_file_list(train_path,dev_path)
        fl_train, fl_val, class_weights_train = train_val_split_file_list(train_path)
        #print (fl_train[:10])
        #print (len(fl_train))


        # Parameters
        params = {'dim': (9,2,125,1),
                'batch_size': 32,
                'n_classes': 2,
                'n_channels': 2,
                'shuffle': True}

        # Datasets
#       partition = # IDs
#       labels = # Labels

        # Generators
        training_generator = DataGenerator(fl_train, **params)
        validation_generator = DataGenerator(fl_val, **params)
        #x,y = training_generator. __getitem__(0)
        #print(y.shape)
        #print(y)


        # Design model
        model = ConvNN(epochs=80)

        #model.setup((-1,1,5,126,20))
        model.setup_3((-1,9,2,125,1))
        # Train model on dataset
        model.fit_5(training_generator, validation_generator,class_weights_train)
        print('finish training and begin evaluate ----------------------------')


    if args.mode == 'test1':
        print('begin generate mode from 3s ,5s and 7s')
        # shallow convlstm
        model = ConvNN(epochs=20)
        # deep  convlstm
        #model = ConvLstmNetDeep(epochs=50)
        evaluate_path = get_path(evaluate_path_all)
        space_time=[3,5,7]

        for j in range(len(space_time)):
            model.setup_3((-1, space_time[j] * 2 - 1, 2, 125, 1))
            model.load_trained_weights(space_time[j])
            dirName = os.path.join('double_check','tempDir_cov_dev__train2_'+str(space_time[j]))
            if not os.path.exists(dirName):
                os.mkdir(dirName)
                print("Directory ", dirName, " Created ")
            else:
                print("Directory ", dirName, " already exists")
            #with open('dev2_files.txt', 'r') as f:
                #while True:

                    # Get next line from file
                    #single_path= f.readline()
                    #single_path = single_path.strip()
                    #print(single_path)
                    # if line is empty
                    # end of file is reached
                    #if not single_path:
                        #break
            for i in range(len(evaluate_path)):
                #result_list=[]
                single_path = evaluate_path[i]

                file_name = single_path.split('/')[-1].split('.')[0]
                data,total_time = get_ref_evalu_stft(single_path,space_time[j])
                result1= model.evaluate_1(data,total_time,space_time[j])
                path = os.path.join(dirName,file_name+'.npy')
                #print(result1)
                print(path)
                np.save(path,result1)
                    #result_list.append(result1)

    if args.mode == 'vote':
        threhold =0.92
        threhold_list=[0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99]
        for k in range(len(threhold_list)):
            threhold=threhold_list[k]
            files3 = glob.glob('/home/yikai/code/competition/szpred_tuh-master 2/results_final/tempDir_cov_eval__train2_3/**/*.npy', recursive=True)
            files5 = glob.glob('/home/yikai/code/competition/szpred_tuh-master 2/results_final/tempDir_cov_eval__train2_5/**/*.npy',recursive=True)
            files7 = glob.glob('/home/yikai/code/competition/szpred_tuh-master 2/results_final/tempDir_cov_eval__train2_7/**/*.npy',recursive=True)
            for i in range(len(files3)):
                data3 = np.load(files3[i])
                #data5 = np.load(files5[i])
                #data7 = np.load(files7[i])
                predictions3 = predict(data3, threhold)
                #predictions5 = predict(data5, threhold)
                #predictions7 = predict(data7, threhold)
                #print(predictions3[:20],predictions5[:20],predictions7[:20])
                file_name = files3[i].split('/')[-1].split('.')[0]
                #vote_wirte(predictions3, predictions5, predictions7, file_name, threhold)
                write(predictions3,file_name, threhold)

        #model.vote_wirte(result_list,file_name)

