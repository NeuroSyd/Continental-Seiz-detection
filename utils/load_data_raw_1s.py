# import os
# import glob
# import datetime
# from datetime import timedelta
import numpy as np
# import pandas as pd
# import scipy.io
from scipy.signal import resample
import stft

# from myio.save_load import save_pickle_file, load_pickle_file, \
#     save_hickle_file, load_hickle_file
# from utils.group_seizure_Kaggle2014Pred import group_seizure

from pyst import read_edf

def calc_stft(s_):
    # s_time = begin_rec + timedelta(seconds=int(i*numts))
    s = s_.transpose()

    stft_data = stft.spectrogram(s,framelength=250,centered=False)
    stft_data = np.transpose(stft_data,(1,2,0))
    stft_data = np.abs(stft_data)+1e-6

    # if self.settings['dataset'] == 'FB':
    #     stft_data = np.concatenate((stft_data[:,:,1:47],
    #                                 stft_data[:,:,54:97],
    #                                 stft_data[:,:,104:]),
    #                                 axis=-1)
    # elif self.settings['dataset'] == 'CHBMIT':
    #     stft_data = np.concatenate((stft_data[:,:,1:57],
    #                                 stft_data[:,:,64:117],
    #                                 stft_data[:,:,124:]),
    #                                 axis=-1)
    # elif self.settings['dataset'] == 'EpilepsiaSurf':
    #     stft_data = stft_data[:,:,1:]
    stft_data = np.log10(stft_data)
    indices = np.where(stft_data <= 0)
    stft_data[indices] = 0


    # from matplotlib import cm
    # from matplotlib import pyplot as plt
    # plt.matshow(stft_data[0]/np.max(stft_data[0]))
    # plt.colorbar()
    # plt.show()

    stft_data = stft_data.reshape(-1, stft_data.shape[0],
                                    stft_data.shape[1],
                                    stft_data.shape[2])


    return stft_data

def get_ref_train_df(ref_train_file, all_files, cachedir):
    window_len = 250 # 1 second
    with open(all_files, 'r') as f:
        all_filenames = f.readlines() 
    print (len(all_filenames))
    # print (all_filenames[-1])
    count = 0
    with open(ref_train_file, 'r') as f:
        while True: 
            count += 1
        
            # Get next line from file 
            line = f.readline() 
        
            # if line is empty 
            # end of file is reached 
            if not line: 
                break
            # print("Line{}: {}".format(count, line.strip())) 
            fn, st, sp, cl, _ = line.strip().split(' ')
            st, sp = float(st), float(sp)
            # print (fn, st, sp, cl)
            fn_full = [name for name in all_filenames if fn in name]
            # print (fn_full)
            if len(fn_full) == 1:
                fn_full = fn_full[0].strip()
                print (fn_full)
                fsamp, data = read_edf(fn_full)
                print (fsamp, data.shape)
                
                # resample to 250 if sampling rate is higher
                if fsamp > 250:
                    print ('Resampling data from {} to 250 Hz'.format(fsamp))
                    data = resample(data, int(data.shape[1]*250/fsamp), axis=1)
                i=0
                while (st+i)*250+window_len < sp*250: 
                    s = data[:, int((st+i)*250) : int((st+i)*250+window_len)]
                    i+=1
                    prep_s = calc_stft(s)
                    prep_fn = '{}/{}_{}_{}.npy'.format(cachedir,fn,i,cl)
                    print ('Raw time-series shape', s.shape)
                    print ('Preprocessed shape', prep_s.shape)
                    np.save(prep_fn, prep_s)
                
               
      

if __name__ == "__main__":   
    ref_train_file = '/mnt/data4/datasets/seizure/TUH_EEG_Seizure/v1.5.1/_DOCS/ref_train.txt'
    all_files = './all_files.txt'
    cachedir = '/mnt/data7_M2/tempdata/tuh_1s_stft'
    get_ref_train_df(ref_train_file, all_files, cachedir)

