# import os
# import glob
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

from pyst import read_edf_elec


def calc_stft(s_):
    # s_time = begin_rec + timedelta(seconds=int(i*numts))
    s = s_.transpose()

    stft_data = stft.spectrogram(s, framelength=250, centered=False)
    stft_data = np.transpose(stft_data, (1, 2, 0))
    stft_data = np.abs(stft_data) + 1e-6

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
    stft_data = stft_data[:,:,1:]
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
    window_len = 250 * 3  # 5 seconds
    with open(all_files, 'r') as f:
        all_filenames = f.readlines()
    print(len(all_filenames))
    # print (all_filenames[-1])
    count = 0
    with open(ref_train_file, 'r') as f:
        while True:

            # Get next line from file
            line = f.readline()

            # if line is empty
            # end of file is reached
            if not line:
                break
            # print("Line{}: {}".format(count, line.strip()))
            fn, st, sp, cl, _ = line.strip().split(' ')
            # if fn == '00010418_s016_t006': # (pyst: nedc_load_edf): failed to open
            #     continue
            count += 1
            st, sp = float(st), float(sp)
            # print (fn, st, sp, cl)
            fn_full = [name for name in all_filenames if fn in name]
            # print (fn_full)
            if len(fn_full) == 1:
                fn_full = fn_full[0].strip()
                print(fn_full)
                fsamp, data = read_edf_elec(fn_full)
                print(fsamp, data.shape)

                # resample to 250 if sampling rate is higher
                if fsamp > 250:
                    print('Resampling data from {} to 250 Hz'.format(fsamp))
                    data = resample(data, int(data.shape[1] * 250.0 / fsamp), axis=1)
                i = 0
                while (st + i) * 250 + window_len < sp * 250:
                    s = data[:, int((st + i) * 250): int((st + i) * 250) + window_len]
                    diff1 = s[3:4, :] - s[2:3, :]
                    # print(i,'done')
                    diff2 = s[13:14, :] - s[17:18, :]
                    s = np.concatenate((diff1, diff2), axis=0)
                    if cl == "seiz":
                        i+=1
                    else:
                        i+=3
                    print('Raw time-series shape', s.shape)

                    assert s.shape[1] == window_len

                    prep_s = calc_stft(s)
                    prep_fn = '{}/{}_{}_{}_{}.npy'.format(cachedir, fn, i, cl, st)

                    print('Preprocessed shape', prep_s.shape)
                    assert prep_s.shape == (1, 5, 2, 125)
                    np.save(prep_fn, prep_s)

                # getting "previous" signals for seizure data
                # take 2 seconds before seizure and concat with the 1st second of sz
                if cl == "seiz":
                    for i_a in range(2):
                        if st - i_a - 1 >= 0:
                            s = data[:, int((st - i_a - 1) * 250): int((st - i_a - 1) * 250) + window_len]
                            diff1 = s[3:4, :] - s[2:3, :]
                            # print(i,'done')
                            diff2 = s[13:14, :] - s[17:18, :]
                            s = np.concatenate((diff1, diff2), axis=0)
                            print('Additional raw time-series shape', st - i_a - 1, s.shape, int((st - i_a - 1) * 250),
                                  int((st - i_a - 1) * 250) + window_len)
                            if s.shape[1] == window_len:
                                # assert s.shape[1] == window_len
                                prep_s = calc_stft(s)
                                prep_fn = '{}/{}_{}_{}_{}.npy'.format(cachedir, fn, -i_a - 1, cl, st)
                                assert prep_s.shape == (1, 5, 2, 125)

                                print('Additional preprocessed shape', prep_s.shape)
                                np.save(prep_fn, prep_s)


if __name__ == "__main__":
    ref_train_file = '/mnt/data4/datasets/seizure/TUH_EEG_Seizure/v1.5.1/_DOCS/ref_train.txt'
    #ref_train_file = '/home/yikai/code/competition/szpred_tuh-master 2/ref_train_1.txt'
    all_files = './all_files.txt'
    cachedir = '/mnt/data7_M2/tempdata/tuh_3s_stft_2ch_O1P3_F3F7_train_whole'
    get_ref_train_df(ref_train_file, all_files, cachedir)
