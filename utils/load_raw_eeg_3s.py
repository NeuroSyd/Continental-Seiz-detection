import numpy as np
# import pandas as pd
# import scipy.io
from scipy.signal import resample
import stft

# from myio.save_load import save_pickle_file, load_pickle_file, \
#     save_hickle_file, load_hickle_file
# from utils.group_seizure_Kaggle2014Pred import group_seizure

from pyst import read_edf


def get_ref_train_df(ref_train_file, all_files, cachedir):
    window_len = 750  # 3 seconds
    with open(all_files, 'r') as f:
        all_filenames = f.readlines()
    print(len(all_filenames))
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
                print(fn_full)
                fsamp, data = read_edf(fn_full)
                print(fsamp, data.shape)

                # resample to 250 if sampling rate is higher
                if fsamp > 250:
                    print('Resampling data from {} to 250 Hz'.format(fsamp))
                    data = resample(data, int(data.shape[1] * 250.0 / fsamp), axis=1)
                i = 0
                while (st + i) * 250 + window_len < sp * 250:
                    s = data[:, int((st + i) * 250): int((st + i) * 250) + window_len]
                    i += 1
                    print('Raw time-series shape', s.shape)

                    assert s.shape[1] == window_len


                    prep_fn = '{}/{}_{}_{}.npy'.format(cachedir, fn, i, cl)

                    #print('Preprocessed shape', prep_s.shape)
                    #assert prep_s.shape == (1, 5, 20, 126)
                    np.save(prep_fn, s)

                # getting "previous" signals for seizure data
                # take 2 seconds before seizure and concat with the 1st second of sz
                if cl == "seiz":
                    for i_a in range(2):
                        if st - i_a - 1 >= 0:
                            s = data[:, int((st - i_a - 1) * 250): int((st - i_a - 1) * 250) + window_len]
                            print('Additional raw time-series shape', st - i_a - 1, s.shape, int((st - i_a - 1) * 250),
                                  int((st - i_a - 1) * 250) + window_len)
                            assert s.shape[1] == window_len
                            #prep_s = calc_stft(s)
                            prep_fn = '{}/{}_{}_{}.npy'.format(cachedir, fn, -i_a - 1, cl)
                            #assert prep_s.shape == (1, 5, 20, 126)

                            print('Additional preprocessed shape', s.shape)
                            np.save(prep_fn, s)


if __name__ == "__main__":
    ref_train_file = '/mnt/data4/datasets/seizure/TUH_EEG_Seizure/v1.5.1/_DOCS/ref_train.txt'
    all_files = './all_files.txt'
    cachedir = '/mnt/data7_M2/tempdata/tuh_3s_raw'
    get_ref_train_df(ref_train_file, all_files, cachedir)
